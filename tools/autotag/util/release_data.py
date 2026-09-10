"""Class to store data about a particular release."""

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from git import Repo
from git.cmd import Git
from github import Github, UnknownObjectException
from github.NamedUser import NamedUser
from github.Organization import Organization
from github.Repository import Repository
from packaging.version import Version

from util.util import get_yn_input
from util.mappings import category_mapping, group_mapping


@dataclass
class ReleaseData:
    """Store Github data for a release."""

    message: str = ""
    notes: str = ""
    changes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReleaseLib:
    """Store data about a release for a particular library."""

    name: str = ""
    repo: Optional[Repository] = None
    pr_repo: Optional[Repository] = None
    data: ReleaseData = field(default_factory=ReleaseData)
    commit: str = ""
    rocm_version: str = ""
    lib_version: str = ""
    group: str = ""
    category: str = ""

    @property
    def qualified_repo(self) -> str:
        """Repo qualified with user/organization."""
        assert self.repo is not None
        return self.repo.full_name

    @property
    def tag(self) -> str:
        """The tag for this release."""
        return f"rocm-{self.full_version}"

    @property
    def branch(self) -> str:
        """The branch for this release."""
        return f"release/rocm-rel-{self.rocm_version}"

    @property
    def full_version(self) -> str:
        """The ROCm full version of this release."""
        return (
            self.rocm_version
            if self.rocm_version.count(".") > 1
            else self.rocm_version + ".0"
        )

    @property
    def release_url(self) -> str:
        """The Github URL of the release."""
        return f"https://github.com/{self.qualified_repo}/releases/tag/{self.tag}"
    
    @property
    def documentation_page(self) -> str:
        """The Read the Docs documentation site."""
        return f"https://rocm.docs.amd.com/projects/{self.qualified_repo}/en/latest"
    
    @property
    def repository_url(self) -> str:
        """The GitHub repository URL."""
        return f"https://github.com/ROCm/{self.qualified_repo}"

    @property
    def message(self) -> str:
        """Get the Github release message."""
        return self.data.message

    @message.setter
    def message(self, value: str):
        """Set the Github release message."""
        self.data.message = value

    @property
    def notes(self) -> str:
        """Get the Github release notes."""
        return self.data.notes

    @notes.setter
    def notes(self, value: str):
        """Set the Github release notes."""
        self.data.notes = value

    def do_release(self, release_yn: Optional[bool]):
        """Perform the tag and release."""
        print(f"Repo: {self.qualified_repo}")
        print(f"Tag Version: '{self.tag}'")
        print(f"Release Message: '{self.data.message}'")
        print(f"Release Notes:\n{self.data.notes}")
        print(f"Release Commit: '{self.commit}'")
        if get_yn_input("Would you like to create this tag and release?", release_yn):
            try:
                print("Performing tag and release.")
                release = self.repo.create_git_tag_and_release(
                    self.tag,
                    self.data.message,
                    self.data.message,
                    self.data.notes,
                    self.commit,
                    "commit",
                )
                if self.rocm_version != self.full_version:
                    self.repo.create_git_tag(
                        f"rocm-{self.rocm_version}",
                        self.data.message,
                        self.commit,
                        "commit",
                    )
                print(release.html_url)
            except Exception:
                print(f"Already released {self.name}")

    def do_create_pull(self, create_pull_yn: Optional[bool], token: str):
        """Create a pull request to the internal repository."""
        if not get_yn_input(
            "Do you want to create a pull request from this release to"
            f" {self.pr_repo.full_name}:develop?",
            create_pull_yn,
        ):
            return
        repo_loc = os.path.join(os.getcwd(), self.name)
        if os.path.isdir(repo_loc):
            shutil.rmtree(repo_loc)

        with Repo.init(repo_loc) as local:
            external = local.create_remote("external", self.repo.clone_url)
            external.fetch()
            fork = local.create_remote(
                "fork",
                f"https://{token}@github.com/"
                f"ROCmMathLibrariesBot/{self.pr_repo.name}",
            )
            fork.fetch()

            local.create_head("release", self.commit).checkout()
            fork.push(f"refs/heads/release:refs/heads/{self.branch}")
        shutil.rmtree(repo_loc)

        pr_title = f"Hotfixes from {self.branch} at release {self.full_version}"
        pr_body = (
            "This is an autogenerated PR.\n This is intended to pull any"
            f" hotfixes for ROCm release {self.full_version} (including"
            " changelogs and documentation) back into develop."
        )
        pr = self.pr_repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=f"ROCmMathLibrariesBot:{self.branch}",
            base="develop",
        )
        print(f"Pull request created: {pr.html_url}")
        return pr


class ReleaseDataFactory:
    """A factory for ReleaseData objects."""

    lib_versions: Dict[str, str] = {}
    """A map of commit hashes to lib versions."""

    def __init__(
        self, org_name: Optional[str], version: str, gh: Github, pr_gh: Github
    ):
        self.gh: Github = gh
        self.pr_gh: Github = pr_gh
        self.rocm_version: Version = version
        if org_name is None:
            self.org = self.pr_org = None
        else:
            self.org, self.pr_org = self.get_org_or_user(org_name)

    def get_org_or_user(
        self, name: str
    ) -> Tuple[Union[NamedUser, Organization], Union[NamedUser, Organization]]:
        """Get a Github organization or user by name."""
        gh_ns: Union[NamedUser, Organization]
        pr_ns: Union[NamedUser, Organization]
        try:
            gh_ns = self.gh.get_organization(name)
            pr_ns = self.pr_gh.get_organization(name)
        except UnknownObjectException:
            try:
                gh_ns = self.gh.get_user(name)
                pr_ns = self.pr_gh.get_user(name)
            except UnknownObjectException as err:
                raise ValueError(f"Could not find organization/user {name}.") from err
        return gh_ns, pr_ns

    def create_release_lib_data(
        self,
        name: str,
        commit: str,
        *,
        org: Optional[str] = None,
    ) -> ReleaseLib:
        """Create a release data object."""
        if self.org is None or self.pr_org is None:
            gh_ns, pr_ns = self.get_org_or_user(org)
        else:
            gh_ns, pr_ns = self.org, self.pr_org
        repo = gh_ns.get_repo(name)
        try:
            pr_repo = pr_ns.get_repo(name + "-internal")
        except UnknownObjectException:
            pr_repo = pr_ns.get_repo(name)
        data = ReleaseLib(
            name=name,
            repo=repo,
            pr_repo=pr_repo,
            commit=commit,
            rocm_version=str(self.rocm_version),
        )
        return data


@dataclass
class ReleaseBundle:
    """Stores data about all the libraries bundled in this release."""

    version: str = ""
    libraries: Dict[str, ReleaseLib] = field(default_factory=dict)


class ReleaseBundleFactory:

    gh: Github = None
    pr_gh: Github = None

    default_remote: str = ""
    """The default fallback remote."""

    remotes: Dict[str, str] = {}
    """A dictionary translating the manifest remote shorthand to the full name."""

    tags: Dict[str, Dict[Version, str]] = {}
    """A dictionary with all the ROCm version numbers and commit sha for each library."""

    orgs_and_users: Dict[
        str, Tuple[Union[NamedUser, Organization], Union[NamedUser, Organization]]
    ] = {}
    """A dictionary containing the base and PR user or organization for each project."""

    pr_repos: Dict[str, Tuple[Repo, Repo]] = {}
    """A dictionary containing the base and PR repo for each project."""

    def __init__(
        self,
        rocm_repo: str,
        gh: Github,
        pr_gh: Github,
        default_remote: str,
        remotes: Dict[str, str],
        branch: Optional[str],
    ):
        # Store Github data
        self.gh = gh
        self.pr_gh = pr_gh

        self.default_remote = default_remote
        self.remotes = remotes
        self.branch = branch

        # Get the main repository:
        self.rocm_repo = gh.get_repo(rocm_repo)

    def get_org(self, remote: str):
        """Find the org associated with the remote, or use the fallback."""
        if remote in self.remotes:
            return self.remotes[remote]
        return self.default_remote

    def get_org_or_user(
        self, remote: str
    ) -> Tuple[Union[NamedUser, Organization], Union[NamedUser, Organization]]:
        """Gets the base and PR organization or user associated to a remote."""
        if remote not in self.orgs_and_users:
            try:
                gh_ns = self.gh.get_organization(remote)
                pr_ns = self.pr_gh.get_organization(remote)
            except UnknownObjectException:
                try:
                    gh_ns = self.gh.get_user(remote)
                    pr_ns = self.pr_gh.get_user(remote)
                except UnknownObjectException as err:
                    raise ValueError(
                        f"Could not find organization/user {remote}."
                    ) from err
            self.orgs_and_users[remote] = (gh_ns, pr_ns)

        return self.orgs_and_users[remote]

    def get_repos(self, name: str, remote: str = None) -> Tuple[Repo, Repo]:
        """Gets the base and PR repository associated to a remote."""
        org = self.get_org(remote)
        if name not in self.pr_repos:
            print(f"Getting remote info for {org}/{name}:")
            gh_ns, pr_ns = self.get_org_or_user(org)
            repo = gh_ns.get_repo(name)
            print(f"  Repo: {repo.url}")
            try:
                pr_repo = pr_ns.get_repo(name + "-internal")
            except UnknownObjectException:
                pr_repo = pr_ns.get_repo(name)
            self.pr_repos[name] = (repo, pr_repo)

        return self.pr_repos[name]

    def get_repo(self, name: str, remote: str) -> Repo:
        """Gets the repository at a remote."""
        path = f"{self.get_org(remote)}/{name}"
        try:
            return self.gh.get_repo(path)
        except Exception as e:
            print(f"Could not get repository {path}.")
            raise e

    def get_tag(self, name: str, version: Version) -> Optional[str]:
        """Finds the Github tag for a library at a ROCm version."""
        if name not in self.tags:
            print(f"Fetching tags for {name}.")
            repo, _ = self.get_repos(name)
            self.tags[name] = self.fetch_tags(repo.clone_url)

        if version not in self.tags[name]:
            return None

        return self.tags[name][version]

    def fetch_tags(self, url: str) -> Dict[Version, str]:
        """Fetches a version-sha map for a given Git URL."""
        result: Dict[Version, str] = {}
        for line in Git().ls_remote("--tags", url).split("\n"):
            column = line.split("\t")
            sha = column[0]
            tag = column[1]

            tag_match = re.search(r"(?P<rocm_tag>rocm-(?P<rocm_ver>\d+(\.\d+)+))", tag)
            if not tag_match:
                continue

            rocm_ver = tag_match["rocm_ver"]
            rocm_ver += ".0" * (2 - rocm_ver.count("."))
            result[Version(rocm_ver)] = sha
        return result

    def create_release_bundle_data(
        self,
        version: Version,
        component_info: List[Tuple[str, str]],
        is_untagged: bool = False,
    ) -> ReleaseBundle:
        """Create a release bundle of libraries."""
        tag_name = f"rocm-{version}"
        libraries = {}

        missing_branches = []

        prev_group = None
        prev_category = None

        print(f"\nLibraries for rocm-{version}:")
        for name, remote, group, category in component_info:
            repo, pr_repo = self.get_repos(name, remote)

            # Find the tag and otherwise
            commit = self.get_tag(name, version)
            if not commit:
                print(f"- Could not find tag '{tag_name}' in '{name}'")
                if not is_untagged:
                    continue

                print(f"  Defaulting to branch: {self.branch}")
                try:
                    repo_branch = repo.get_branch(self.branch)
                    commit = repo_branch.commit.sha
                except Exception:
                    print(f"  - Could not find branch : {self.branch}")
                    missing_branches.append(f"{self.branch} for {name}")
                    continue

            if prev_group == group:
                group = ""
            else:
                prev_group = group

            if prev_category == category:
                category = ""
            else:
                prev_category = category

            libraries[name] = ReleaseLib(
                name=name,
                repo=repo,
                pr_repo=pr_repo,
                commit=commit,
                rocm_version=str(version),
                group=group_mapping[group],
                category=category_mapping[category],
            )

            print(f"- {name:11} {commit}")

        data = ReleaseBundle(version=version, libraries=libraries)

        for missing in missing_branches:
            print(f"Could not find the following branch: {missing}")

        return data

    def create_data_dict(
        self,
        up_to_version: str,
        component_information: List[Tuple[str, str]],
        min_version: str = "5.0.0",
    ) -> Dict[str, ReleaseBundle]:
        """Create a map of versions and release bundles."""

        # Get the tags and versions
        max_version = Version(up_to_version)
        rocm_tags = self.fetch_tags(self.rocm_repo.clone_url)
        versions = list(rocm_tags.keys())

        if up_to_version not in versions:
            versions.append(max_version)
        versions.sort()

        # For each ROCm release, create a bundle.
        data = {}
        for version in versions:
            if version >= Version(min_version) and version <= max_version:
                can_be_untagged = version == max_version
                data[str(version)] = self.create_release_bundle_data(
                    version, component_information, can_be_untagged
                )

        return data

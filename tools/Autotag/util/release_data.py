"""Class to store data about a particular release."""

from dataclasses import dataclass, field
import os
import shutil
from typing import Optional, Union
from github import Github, UnknownObjectException
from github.Repository import Repository
from github.Organization import Organization
from github.NamedUser import NamedUser
from git import Repo

from util.util import get_yn_input


@dataclass
class ReleaseData:
    """Store Github data for a release."""

    message: str = ""
    notes: str = ""


@dataclass
class ReleaseLib:
    """Store data about a release for a particular library."""

    name: str = ""
    repo: Optional[Repository] = None
    pr_repo: Optional[Repository] = None
    data: ReleaseData = field(default_factory=ReleaseData)
    commit: str = ""
    rocm_version: str = ""

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
        if get_yn_input(
            "Would you like to create this tag and release?", release_yn
        ):
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

        pr_title = (
            f"Hotfixes from {self.branch} at release {self.full_version}"
        )
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

    def __init__(
        self, org_name: Optional[str], version: str, gh: Github, pr_gh: Github
    ):
        self.gh: Github = gh
        self.pr_gh: Github = pr_gh
        self.rocm_version: str = version
        if org_name is None:
            self.org = self.pr_org = None
        else:
            self.org, self.pr_org = self.get_org_or_user(org_name)

    def get_org_or_user(self, name: str):
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
                raise ValueError(
                    f"Could not find organization/user {name}."
                ) from err
        return gh_ns, pr_ns

    def create_data(
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
            rocm_version=self.rocm_version,
        )
        return data

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
import requests
import re

logger = logging.getLogger(__name__)

class BranchAwareRemoteContent(Directive):
    """
    Directive that downloads and includes content from other repositories,
    matching the branch/tag of the current documentation build.

    Usage:
    .. remote-content::
       :repo: owner/repository
       :path: path/to/file.rst
       :default_branch: docs/develop  # Branch to use when not on a release
       :tag_prefix: Docs/  # Optional
    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = False
    option_spec = {
        'repo': str,
        'path': str,
        'default_branch': str,  # Branch to use when not on a release tag
        'start_line': int,      # Include the file from a specific line
        'tag_prefix': str,      # Prefix for release tags (e.g., 'Docs/')
    }

    def get_current_version(self):
        """Get current version/branch being built"""
        env = self.state.document.settings.env
        html_context = env.config.html_context

        # Check if building from a tag
        if "official_branch" in html_context:
            if html_context["official_branch"] == 0:
                if "version" in html_context:
                    # Remove any 'v' prefix
                    version = html_context["version"]
                    if re.match(r'^\d+\.\d+\.\d+$', version):
                        return version

        # Not a version tag, so we'll use the default branch
        return None

    def get_target_ref(self):
        """Get target reference for the remote repository"""
        current_version = self.get_current_version()

        # If it's a version number, use tag prefix and version
        if current_version:
            tag_prefix = self.options.get('tag_prefix', '')
            return f'{tag_prefix}{current_version}'

        # For any other case, use the specified default branch
        if 'default_branch' not in self.options:
            logger.warning('No default_branch specified and not building from a version tag')
            return None

        return self.options['default_branch']

    def construct_raw_url(self, repo, path, ref):
        """Construct the raw.githubusercontent.com URL"""
        return f'https://raw.githubusercontent.com/{repo}/{ref}/{path}'

    def fetch_and_parse_content(self, url, source_path):
        """Fetch content and parse it as RST"""
        response = requests.get(url)
        response.raise_for_status()
        content = response.text

        start_line = self.options.get('start_line', 0)

        # Create ViewList for parsing
        line_count = 0
        content_list = ViewList()
        for line_no, line in enumerate(content.splitlines()):
            if line_count >= start_line:
                content_list.append(line, source_path, line_no)
            line_count+=1 

        # Create a section node and parse content
        node = nodes.section()
        nested_parse_with_titles(self.state, content_list, node)

        return node.children

    def run(self):
        if 'repo' not in self.options or 'path' not in self.options:
            logger.warning('Both repo and path options are required')
            return []

        target_ref = self.get_target_ref()
        if not target_ref:
            return []

        raw_url = self.construct_raw_url(
            self.options['repo'],
            self.options['path'],
            target_ref
        )

        try:
            logger.info(f'Attempting to fetch content from {raw_url}')
            return self.fetch_and_parse_content(raw_url, self.options['path'])
        except requests.exceptions.RequestException as e:
            logger.warning(f'Failed to fetch content from {raw_url}: {str(e)}')

            # If we failed on a tag, try falling back to default_branch
            if re.match(r'^\d+\.\d+\.\d+$', target_ref) or target_ref.startswith('Docs/'):
                if 'default_branch' in self.options:
                    try:
                        fallback_ref = self.options['default_branch']
                        logger.info(f'Attempting fallback to {fallback_ref}...')

                        fallback_url = self.construct_raw_url(
                            self.options['repo'],
                            self.options['path'],
                            fallback_ref
                        )

                        return self.fetch_and_parse_content(fallback_url, self.options['path'])
                    except requests.exceptions.RequestException as e2:
                        logger.warning(f'Fallback also failed: {str(e2)}')

            return []

def setup(app):
    app.add_directive('remote-content', BranchAwareRemoteContent)

    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

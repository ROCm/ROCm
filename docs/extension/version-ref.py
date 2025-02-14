from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging
import csv
from io import StringIO
import re
import shlex

logger = logging.getLogger(__name__)

class VersionReference(nodes.Inline, nodes.TextElement):
    """Represents an inline version reference."""
    pass

class VersionSetDirective(Directive):
    """Directive for setting version references within a page scope."""
    required_arguments = 2  # name and value
    optional_arguments = 0

    def run(self):
        env = self.state.document.settings.env
        if not hasattr(env, 'doc_version_refs'):
            env.doc_version_refs = {}
        current_doc = env.docname
        if current_doc not in env.doc_version_refs:
            env.doc_version_refs[current_doc] = {}

        name, value = self.arguments
        if name.lower() == 'latest':
            logger.warning('Cannot override the "latest" keyword with version-set')
            return []

        # Handle 'latest' value by getting the actual version
        if value.lower() == 'latest':
            data = getattr(env, 'compatibility_matrix', None)
            if data:
                latest_version = get_latest_rocm_version(data)
                if latest_version:
                    value = latest_version

        env.doc_version_refs[current_doc][name] = value
        return []

def clean_library_name(name):
    """Extract library name from RST formatting."""
    # Handle :doc: format
    doc_match = re.search(r':doc:`([^<]+)(?:\s+<[^>]+>)?`', name)
    if doc_match:
        return doc_match.group(1).strip()

    # Handle other link formats
    link_match = re.search(r'`([^<]+)(?:\s+<[^>]+>)?`_?', name)
    if link_match:
        return link_match.group(1).strip()

    return name.strip()

def get_latest_rocm_version(data):
    """Get the latest ROCm version from the matrix headers."""
    if not data or len(data) == 0:
        return None

    # Get all column names except 'ROCm Version'
    columns = [col for col in data[0].keys() if col != 'ROCm Version']
    # Return the first column name (assumed to be the latest version)
    return columns[0] if columns else None

def version_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Role function to print version value.
    Usage: :version:`version_name`
    """
    try:
        version_name = text.strip()
        env = inliner.document.settings.env

        if hasattr(env, 'doc_version_refs'):
            current_doc = env.docname
            if current_doc in env.doc_version_refs:
                doc_refs = env.doc_version_refs[current_doc]
                if version_name in doc_refs:
                    version = doc_refs[version_name]
                    node = nodes.Text(version)
                    return [node], []

        msg = inliner.reporter.warning(
            f'No version defined for name {version_name}',
            line=lineno
        )
        return [], [msg]

    except Exception as e:
        msg = inliner.reporter.error(
            f'Error looking up version: {str(e)}',
            line=lineno
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

def version_ref_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Role function for version references.
    Usage: :version-ref:`library_name release`
           :version-ref:`"library name" release`
           :version-ref:`library_name latest`
           :version-ref:`rocm latest`
    """
    try:
        # Parse the text - handle both quoted and unquoted formats
        if '"' in text:
            parts = shlex.split(text)
        else:
            parts = text.split()

        if len(parts) != 2:
            msg = inliner.reporter.error(
                'Version reference must be in format "library_name release" or "\\"library name\\" release"',
                line=lineno
            )
            prb = inliner.problematic(rawtext, rawtext, msg)
            return [prb], [msg]

        library_name, release = parts
        env = inliner.document.settings.env

        # Check if release is a version reference in current document
        if hasattr(env, 'doc_version_refs'):
            current_doc = env.docname
            if current_doc in env.doc_version_refs:
                doc_refs = env.doc_version_refs[current_doc]
                if release in doc_refs:
                    release = doc_refs[release]

        # Handle special case for "rocm latest"
        if library_name.lower() == 'rocm' and release.lower() == 'latest':
            data = getattr(env, 'compatibility_matrix', None)
            if not data:
                raise ValueError("Compatibility matrix not found in environment")

            latest_version = get_latest_rocm_version(data)
            if latest_version:
                node = VersionReference()
                node += nodes.Text(latest_version)
                return [node], []
            else:
                msg = inliner.reporter.warning(
                    'No ROCm versions found in compatibility matrix',
                    line=lineno
                )
                return [], [msg]

        version = lookup_version(inliner, library_name, release)

        if version:
            node = VersionReference()
            node += nodes.Text(version)
            return [node], []
        else:
            msg = inliner.reporter.warning(
                f'No version found for library {library_name} in release {release}',
                line=lineno
            )
            return [], [msg]

    except Exception as e:
        msg = inliner.reporter.error(
            f'Error looking up version: {str(e)}',
            line=lineno
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

def lookup_version(inliner, library_name, release):
    """Look up the version in the compatibility matrix."""
    env = inliner.document.settings.env
    data = getattr(env, 'compatibility_matrix', None)

    if not data:
        raise ValueError("Compatibility matrix not found in environment")

    # Handle the 'latest' keyword
    if release.lower() == 'latest':
        latest_version = get_latest_rocm_version(data)
        if not latest_version:
            return None
        release = latest_version

    # For ROCm, check if the version exists in column headers
    if library_name.lower() == 'rocm':
        columns = [col for col in data[0].keys() if col != 'ROCm Version']
        if release in columns:
            return release
        return None

    # Find the library version
    for row in data:
        row_lib_name = clean_library_name(row['ROCm Version'])
        if row_lib_name == library_name:
            # Get the version, removing any whitespace
            version = row.get(release, '').strip()
            if version:
                return version

    # If not found, try a case-insensitive search
    for row in data:
        row_lib_name = clean_library_name(row['ROCm Version'])
        if row_lib_name.lower() == library_name.lower():
            version = row.get(release, '').strip()
            if version:
                return version

    return None

def visit_version_reference(self, node):
    self.body.append(f'<span class="version-reference">')

def depart_version_reference(self, node):
    self.body.append('</span>')

def load_compatibility_matrix(app):
    """Load the compatibility matrix content from CSV."""
    if not app.config.compatibility_matrix_file:
        logger.warning('No compatibility matrix file configured')
        return

    try:
        with open(app.config.compatibility_matrix_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            app.env.compatibility_matrix = list(reader)
            logger.info('Successfully loaded compatibility matrix')

            # Debug: print first few rows with their library names
            for row in list(app.env.compatibility_matrix)[:5]:
                if 'ROCm Version' in row:
                    lib_name = clean_library_name(row['ROCm Version'])
                    logger.debug(f"Loaded library: {lib_name}")

    except Exception as e:
        logger.error(f'Error loading compatibility matrix: {str(e)}')

def purge_version_refs(app, env, docname):
    """Remove version references for a document when it is purged"""
    if hasattr(env, 'doc_version_refs'):
        if docname in env.doc_version_refs:
            del env.doc_version_refs[docname]

def setup(app):
    app.add_node(VersionReference,
                 html=(visit_version_reference, depart_version_reference))
    app.add_role('version-ref', version_ref_role)
    app.add_role('version', version_role)
    app.add_directive('version-set', VersionSetDirective)

    # Add a config value for the compatibility matrix file path
    app.add_config_value('compatibility_matrix_file', None, 'env')

    # Connect to the builder-inited event to load the matrix
    app.connect('builder-inited', load_compatibility_matrix)

    # Connect to env-purge-doc event to clean up document-specific version refs
    app.connect('env-purge-doc', purge_version_refs)

    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

from core.utils import io_worker as iw

DIR_ROOT = "ROOT_DIR_PLACEHOLDER"  # Replace with actual root directory path
# Version of the dumps
DUMPS_VERSION_WP_HTML = "20220301"
DIR_MNT = "MOUNT_DIR_PLACEHOLDER"  # Replace with actual mount directory path
# Configuration
ENCODING = "utf-8"

# Directories
DIR_DUMPS = f"{DIR_ROOT}/data/dump"
DIR_MODELS = f"{DIR_ROOT}/data/models"
DIR_CONFIG = f"{DIR_ROOT}/config"
DIR_WIKI_TABLES = f"{DIR_MNT}/wikitables/final_data"

# 322 languages of Wikipedia
LANGS = iw.read_tsv_file_first_col(f"{DIR_CONFIG}/LANGS_322.tsv", ENCODING)

HTML_HEADERS = iw.read_tsv_file_first_col(
    f"{DIR_CONFIG}/TAGS_HTML_HEADERS.tsv", ENCODING
)

URL_WP_HTML = "https://dumps.wikimedia.org/other/enterprise_html/runs/{wikipedia_version}/{lang}wiki-NS0-{wikipedia_version}-ENTERPRISE-HTML.json.tar.gz"

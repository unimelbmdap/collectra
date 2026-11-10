These pinned third-party assets are bundled so the GUI can run without network access.
See manifest.json for package versions, upstream archive URLs, licenses, and included files.
Each package's license is kept in its directory; OpenSeadragon is in ../openseadragon.

Local changes:
- Bootswatch's Google Fonts import is replaced with a comment. index.html loads
  the bundled Source Sans Pro stylesheet instead (400, 600, and 700 weights).
- Source Sans Pro's stylesheet combines its three normal Latin font faces.
- Only assets used by the GUI and their license files are included.

EasyMDE is configured with autoDownloadFontAwesome: false and spellChecker: false
so it does not fetch additional dependencies at runtime.

Bootstrap's `scss/` folder is bundled unmodified so that a pipeline's
`theme: FILENAME.scss` can be compiled at load time (see collectra/gui/scss_theme.py).

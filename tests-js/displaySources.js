// Source-inspection tests include the external display implementations too.
const fs = require('node:fs');
const path = require('node:path');
const gui = path.resolve(__dirname, '../collectra/gui');
module.exports = ['index.html', 'displays/image.js', 'displays/imageCrop.js',
    'displays/imageSegmentation.js', 'displays/text.js', 'displays/properties.js']
    .map(file => fs.readFileSync(path.join(gui, file), 'utf8')).join('\n');

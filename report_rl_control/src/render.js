// usage: node render.js out_dir stem1 stem2 ...  -> stem.png (3x)
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [dir, ...stems] = process.argv.slice(2);
  const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
  const page = await browser.newPage({ deviceScaleFactor: 3 });
  for (const s of stems) {
    const svg = fs.readFileSync(`${dir}/${s}.svg`, 'utf8');
    await page.setContent(`<html><body style="margin:0">${svg}</body></html>`);
    const el = await page.$('svg');
    await el.screenshot({ path: `${dir}/${s}.png` });
  }
  await browser.close();
})();

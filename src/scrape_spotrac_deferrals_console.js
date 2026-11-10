/*
 * Spotrac deferred-money scraper (browser console, any https://www.spotrac.com/mlb/... page).
 * Downloads spotrac_deferrals_raw.csv ('|' separated): all-time contracts with deferred money.
 * Columns: spotrac_id|years|total|orig_years|deferred|def_years   (money in dollars; years as YY-YY)
 * Feeds src/adjusted_aav.py. The pieces were run in a browser on 2026-11-10; the packaged file is not yet re-run end to end.
 */
(async () => {
  const h = await fetch('/mlb/contracts/deferred/_/type/all-time').then(r => r.text());
  const t = new DOMParser().parseFromString(h, 'text/html').querySelector('table');
  const rows = [...t.rows].slice(1).map(tr => {
    const c = [...tr.cells].map(x => x.textContent.trim().replace(/\s+/g, ' '));
    const id = ((tr.querySelector('a[href*="/player/"]') || {}).href || '').match(/id\/(\d+)/);
    const m = c[3].match(/(\d+) yr, \$([\d,]+)/);
    return [id ? id[1] : '', m ? m[1] : '', m ? m[2].replace(/,/g, '') : '', c[4].replace(/20(\d\d)-20(\d\d)/, '$1-$2'),
            c[5].replace(/[$,]/g, ''), c[7].replace(/20(\d\d)-20(\d\d)/, '$1-$2')].join('|');
  });
  const csv = ['spotrac_id|years|total|orig_years|deferred|def_years'].concat(rows).join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
  a.download = 'spotrac_deferrals_raw.csv'; document.body.appendChild(a); a.click(); a.remove();
  console.log(`done: ${rows.length} deferral rows`);
})();

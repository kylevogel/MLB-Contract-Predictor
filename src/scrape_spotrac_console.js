/*
 * Spotrac scraper (browser console). Run on any https://www.spotrac.com/mlb/... page while
 * the page is open (same origin, so fetch() works). Writes spotrac_raw_contracts.csv.
 * Polite: ~0.5 s between requests, ~60 requests total.
 *
 * URL patterns used (discovered Nov 2026; Spotrac ignores ?query params, it needs path form):
 *   Extensions with dates : /mlb/transactions/_/start/2011-01-01/end/<today>/page/N/type/signed-extended
 *   Extensions by type    : /mlb/contracts/extensions/_/type/{pre-arbitration-extension|arbitration-extension|extension}
 *   Extensions by year    : /mlb/contracts/extensions/_/year/YYYY   (YYYY = contract start year)
 *   Free agents by year   : /mlb/free-agents/signed/_/year/YYYY     (YYYY = first season of the contract)
 * Class rule: offseason = YYYY-1 for free agents; for extensions use the transaction date when present.
 */
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const parse = h => new DOMParser().parseFromString(h, 'text/html');
  const MON = {Jan:1,Feb:2,Mar:3,Apr:4,May:5,Jun:6,Jul:7,Aug:8,Sep:9,Oct:10,Nov:11,Dec:12};
  const money = s => +s.replace(/[$,]/g, '') / 1e6;
  const today = new Date().toISOString().slice(0, 10);

  async function table(path, nameCol) {
    const tb = parse(await fetch(path).then(r => r.text())).querySelector('table');
    if (!tb) return [];
    return [...tb.rows].slice(1).map(tr => {
      const c = [...tr.cells].map(x => x.textContent.trim().replace(/\s+/g, ' '));
      const m = (tr.querySelector('a[href*="/player/"]') || {}).href?.match(/id\/(\d+)\/([^/?#]+)/);
      return m ? {c, id: +m[1], slug: m[2]} : null;
    }).filter(Boolean);
  }

  // 1) extension transactions (give dates)
  const tx = [], seen = new Set();
  for (let p = 1; p <= 30; p++) {
    const d = parse(await fetch(`/mlb/transactions/_/start/2011-01-01/end/${today}/page/${p}/type/signed-extended`).then(r => r.text()));
    const rows = [...d.querySelectorAll('li.transaction-row')];
    if (!rows.length) break;
    for (const li of rows) {
      const dt = li.querySelector('.transaction-date').textContent.match(/([A-Z][a-z]{2}) (\d{2}), (\d{4})/);
      const a = li.querySelector('.transaction-details a[href*="/player/"]');
      const desc = (li.querySelector('.transaction-details small') || {}).textContent || '';
      const m = a && a.href.match(/id\/(\d+)\/([^/?#]+)/);
      const vm = desc.match(/\$([\d.,]+) (million|billion)/), ym = desc.match(/(\d+) year/);
      const rec = {date: dt ? `${dt[3]}-${String(MON[dt[1]]).padStart(2,'0')}-${dt[2]}` : null, id: m ? +m[1] : null, desc,
                   yrs: ym ? +ym[1] : null, valM: vm ? parseFloat(vm[1].replace(/,/g,'')) * (vm[2] === 'billion' ? 1000 : 1) : null};
      const k = [rec.date, rec.id, desc].join('|');
      if (rec.id && !seen.has(k)) { seen.add(k); tx.push(rec); }   // page past the end repeats rows -> dedupe
    }
    await sleep(500);
  }

  // 2) extensions by type, then by start year
  const ext = [];
  for (const [type, slug] of [['extension_pre_arb','pre-arbitration-extension'],['extension_arb','arbitration-extension'],['extension_veteran','extension']]) {
    (await table(`/mlb/contracts/extensions/_/type/${slug}`)).forEach(r =>
      ext.push({type, id: r.id, slug: r.slug, name: r.c[1], pos: r.c[2], team: r.c[3], age: +r.c[4], yrs: +r.c[5], val: money(r.c[6]), aav: money(r.c[7])}));
    await sleep(400);
  }
  const yearOf = new Map();
  for (let y = 2013; y <= 2027; y++) {
    (await table(`/mlb/contracts/extensions/_/year/${y}`)).forEach(r => yearOf.set([r.id, +r.c[5], Math.round(money(r.c[6]) * 100)].join('|'), y));
    await sleep(400);
  }
  ext.forEach(e => e.year = yearOf.get([e.id, e.yrs, Math.round(e.val * 100)].join('|')) || null);

  // 3) join dates onto extensions: exact (id, years, value +-0.6M), then unique-by-id fallback
  const used = new Set(), out = [];
  const take = (a, loose) => {
    const c = ext.filter((b, i) => !used.has(i) && b.id === a.id && (loose ? (!a.yrs || b.yrs === a.yrs)
              : b.yrs === a.yrs && Math.abs(b.val - a.valM) < 0.6));
    if (c.length >= (loose ? 1 : 1) && (!loose || c.length === 1)) { used.add(ext.indexOf(c[0])); out.push({...c[0], date: a.date, date_source: 'transactions'}); }
  };
  tx.filter(a => a.yrs && a.valM).forEach(a => take(a, false));
  tx.filter(a => !(a.yrs && a.valM)).forEach(a => take(a, true));
  ext.forEach((b, i) => { if (!used.has(i)) out.push({...b, date: null, date_source: 'signed_year'}); });

  // 4) free agents by start year (2015..2026 = offseasons 2014..2025)
  const fa = [];
  for (let y = 2015; y <= 2026; y++) {
    (await table(`/mlb/free-agents/signed/_/year/${y}`)).forEach(r => fa.push({year: y, id: r.id, slug: r.slug,
      name: r.c[3], from: r.c[0], to: r.c[2], pos: r.c[4], yrs: +r.c[5], val: money(r.c[6]), aav: money(r.c[7])}));
    await sleep(500);
  }

  // 5) CSV
  const q = s => /[",\n]/.test(String(s)) ? '"' + String(s).replace(/"/g, '""') + '"' : String(s);
  const cols = ['source','contract_type','spotrac_id','slug','player_name','position','team_from','team_to','age_at_signing',
                'years_new','total_value_m','aav_nominal_m','signing_date','date_source','spotrac_start_year','qualifying_offer','is_zero_value'];
  const rows = [
    ...fa.map(f => ['free_agents_page','free_agent',f.id,f.slug,f.name.replace(/ QO$/,''),f.pos,f.from,f.to,'',f.yrs,f.val,f.aav,'','none',f.year,/ QO$/.test(f.name)?1:0,f.val===0?1:0]),
    ...out.map(e => ['extensions_page',e.type,e.id,e.slug,e.name,e.pos,e.team,e.team,e.age,e.yrs,e.val,e.aav,e.date||'',e.date_source,e.year||'',0,0])];
  const csv = [cols.join(',')].concat(rows.map(r => r.map(q).join(','))).join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
  a.download = 'spotrac_raw_contracts.csv'; document.body.appendChild(a); a.click(); a.remove();
  console.log(`done: ${fa.length} FA rows, ${out.length} extension rows`);
})();

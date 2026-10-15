/* 챔피언스 배틀 AI — 화면 */
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style') el.style.cssText = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(c));
  return el;
}
const api = async (url, body) => {
  const r = await fetch(url, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || '오류');
  return j;
};
const toast = (msg) => { const t = $('#toast'); t.textContent = msg; t.classList.add('on'); clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove('on'), 2400); };
const store = {
  get(k, d) { try { const v = localStorage.getItem('pba:' + k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('pba:' + k, JSON.stringify(v)); } catch (e) {} },
};
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const sprite = (sid, cls = '') => h('img', { class: 'sprite ' + cls, src: '/sprite/' + (sid || 'unknown'), loading: 'lazy', alt: '' });
const pct = (x) => x == null ? '–' : Math.round(x * 100) + '%';

const STAT_KO = { hp: 'HP', atk: '공격', def: '방어', spa: '특공', spd: '특방', spe: '스피드' };
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const BOOST_KEYS = ['atk', 'def', 'spa', 'spd', 'spe'];
const STATUS = [['', '정상'], ['brn', '화상'], ['par', '마비'], ['psn', '독'], ['tox', '맹독'], ['slp', '잠듦'], ['frz', '얼음']];
const WEATHER = [['', '없음'], ['sunnyday', '쾌청'], ['raindance', '비'], ['sandstorm', '모래바람'], ['snowscape', '설경']];
const TERRAIN = [['', '없음'], ['electricterrain', '일렉트릭필드'], ['grassyterrain', '그래스필드'], ['psychicterrain', '사이코필드'], ['mistyterrain', '미스트필드']];
const SIDE = [['reflect', '리플렉터'], ['lightscreen', '빛의장막'], ['auroraveil', '오로라베일'], ['tailwind', '순풍'], ['stealthrock', '스텔스록'], ['spikes', '압정뿌리기'], ['toxicspikes', '독압정'], ['stickyweb', '끈적끈적네트']];

// ======================================================================== 상태
const S = {
  meta: null, gt: store.get('gt', 'singles'),
  team: store.get('team', { id: '', name: '', team: [] }),
  oppPreview: store.get('oppPreview', []),
  battle: store.get('battle', null),
  names: {},
  advice: null,
};
const typesKo = (t) => (S.meta && S.meta.typesKo[t]) || t;
const typeBadge = (t) => h('span', { class: 'type t-' + t }, typesKo(t));
function saveState() { store.set('gt', S.gt); store.set('team', S.team); store.set('oppPreview', S.oppPreview); store.set('battle', S.battle); }

// 이름 캐시 (한글)
const NAME = { species: {}, moves: {}, items: {}, abilities: {} };
async function ensureNames(req) {
  const need = {};
  for (const [k, ids] of Object.entries(req)) {
    const miss = [...new Set(ids.filter(i => i && !(i in NAME[k])))];
    if (miss.length) need[k] = miss;
  }
  if (!Object.keys(need).length) return;
  const r = await api('/api/names', need);
  for (const [k, m] of Object.entries(r.names)) Object.assign(NAME[k], m);
}
const nm = (k, id) => (id && NAME[k][id]) || id || '';

// ======================================================================== 자동완성
function autocomplete({ kind, value = '', placeholder = '', species = null, onPick, showSprite = kind === 'species', noMega = false }) {
  const wrap = h('div', { class: 'ac' });
  const inp = h('input', { type: 'text', placeholder, value: value || '' });
  const list = h('div', { class: 'list' });
  wrap.append(inp, list);
  let items = [], sel = -1;
  const load = debounce(async () => {
    const q = inp.value.trim();
    const params = new URLSearchParams({ kind, q, limit: 30 });
    if (species) params.set('species', typeof species === 'function' ? species() : species);
    if (noMega) params.set('noMega', '1');
    try {
      const r = await api('/api/search?' + params);
      items = r.results; sel = -1; render();
    } catch (e) { list.classList.remove('on'); }
  }, 120);
  function render() {
    list.innerHTML = '';
    items.forEach((it, i) => {
      const meta = kind === 'moves' ? `${it.basePower || '-'} · ${it.category === 'Physical' ? '물리' : it.category === 'Special' ? '특수' : '변화'}` : '';
      const row = h('div', { class: 'it' + (i === sel ? ' sel' : ''), onmousedown: (e) => { e.preventDefault(); pick(it); } },
        showSprite ? h('img', { src: '/sprite/' + it.id, loading: 'lazy' }) : null,
        h('span', {}, it.ko), h('span', { class: 'muted small' }, it.name),
        kind === 'species' ? h('span', { class: 'meta' }, ...(it.types || []).map(typeBadge)) :
          kind === 'moves' ? h('span', { class: 'meta' }, typeBadge(it.type), ' ', meta) : null);
      list.append(row);
    });
    list.classList.toggle('on', items.length > 0 && document.activeElement === inp);
  }
  function pick(it) { inp.value = it.ko; list.classList.remove('on'); NAME[kind][it.id] = it.ko; onPick && onPick(it); }
  inp.addEventListener('input', load);
  inp.addEventListener('focus', () => { inp.select(); load(); });
  inp.addEventListener('blur', () => setTimeout(() => list.classList.remove('on'), 120));
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { sel = Math.min(items.length - 1, sel + 1); render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { sel = Math.max(0, sel - 1); render(); e.preventDefault(); }
    else if (e.key === 'Enter' && items[Math.max(0, sel)]) { pick(items[Math.max(0, sel)]); e.preventDefault(); }
  });
  wrap.input = inp;
  return wrap;
}
function selectEl(options, value, onChange) {
  const s = h('select', { onchange: (e) => onChange(e.target.value) }, ...options.map(([v, t]) => h('option', { value: v, selected: v === value }, t)));
  return s;
}

// ======================================================================== 탭
function showPage(p) {
  $$('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.p === p));
  $$('.page').forEach(s => s.classList.toggle('on', s.id === 'p-' + p));
  store.set('page', p);
  if (p === 'ai') renderAI();
  if (p === 'settings') renderSettings();
  if (p === 'battle') renderBattle();
  if (p === 'bring') renderBring();
  if (p === 'calc') renderCalc();
  if (p === 'build') renderBuild();
  if (p === 'play') renderPlay();
}
$$('#tabs button').forEach(b => b.addEventListener('click', () => showPage(b.dataset.p)));
$$('#gtSeg button').forEach(b => b.addEventListener('click', () => {
  S.gt = b.dataset.gt; $$('#gtSeg button').forEach(x => x.classList.toggle('on', x === b));
  saveState(); if (S.battle) { S.battle.gameType = S.gt; } renderBattle(); renderBring();
}));

// ======================================================================== 팀
const emptySet = () => ({ species: '', ability: '', item: '', nature: 'serious', sp: {}, moves: [] });
async function renderTeamList() {
  const r = await api('/api/team/list');
  const el = $('#teamList'); el.innerHTML = '';
  if (!r.teams.length) el.append(h('div', { class: 'empty small' }, '저장한 팀이 없습니다.'));
  for (const t of r.teams) {
    el.append(h('div', { class: 't' + (t.id === S.team.id ? ' on' : ''), onclick: () => { S.team = { id: t.id, name: t.name, team: t.team }; saveState(); renderTeam(); renderTeamList(); updatePill(); } },
      h('div', { class: 'col', style: 'flex:1; gap:2px' }, h('b', {}, t.name), h('div', { class: 'row', style: 'gap:0' }, ...t.team.map(s => h('img', { src: '/sprite/' + s.species, loading: 'lazy' })))),
      h('button', { class: 'btn sm ghost', onclick: async (e) => { e.stopPropagation(); if (!confirm(t.name + ' 팀을 지울까요?')) return; await api('/api/team/delete', { id: t.id }); renderTeamList(); } }, '삭제')));
  }
}
function updatePill() {
  const n = (S.team.team || []).filter(s => s.species).length;
  $('#teamPill').textContent = S.team.name ? `${S.team.name} · ${n}마리` : (n ? `이름 없는 팀 · ${n}마리` : '팀 없음');
}
async function renderTeam() {
  const team = S.team.team;
  while (team.length < 6) team.push(emptySet());
  await ensureNames({
    species: team.map(s => s.species), items: team.map(s => s.item), abilities: team.map(s => s.ability),
    moves: team.flatMap(s => s.moves || []),
  });
  $('#teamName').value = S.team.name || '';
  const el = $('#slots'); el.innerHTML = '';
  let stats = [];
  try { stats = (await api('/api/team/stats', { team: team.filter(s => s.species) })).stats; } catch (e) {}
  let si = 0;
  team.forEach((s, i) => el.append(slotCard(s, i, s.species ? stats[si++] : null)));
  checkTeam();
  updatePill();
}
function checkTeam() {
  const t = S.team.team.filter(s => s.species);
  const items = t.map(s => s.item).filter(Boolean);
  const dupItem = items.filter((x, i) => items.indexOf(x) !== i);
  const bases = t.map(s => s.species.replace(/mega.*$/, ''));
  const dupSp = bases.filter((x, i) => bases.indexOf(x) !== i);
  const msgs = [];
  if (dupItem.length) msgs.push('같은 도구 중복: ' + [...new Set(dupItem)].map(i => nm('items', i)).join(', '));
  if (dupSp.length) msgs.push('같은 포켓몬 중복');
  t.forEach(s => { const tot = Object.values(s.sp || {}).reduce((a, b) => a + (+b || 0), 0); if (tot > 66) msgs.push(`${nm('species', s.species)} SP ${tot}/66 초과`); });
  $('#teamCheck').innerHTML = msgs.length ? '<span style="color:var(--bad)">⚠ ' + msgs.join(' · ') + '</span>' : (t.length ? `${t.length}마리 · 규칙 확인 완료` : '');
}
function slotCard(s, i, st) {
  const card = h('div', { class: 'card slot' });
  const changed = () => { saveState(); checkTeam(); updatePill(); };
  const rerender = () => { saveState(); renderTeam(); };
  const spAc = autocomplete({ kind: 'species', value: nm('species', s.species), placeholder: '포켓몬 검색', noMega: true, onPick: async (it) => {
    s.species = it.id; s.moves = []; s.ability = '';
    try { const r = await api('/api/species/' + it.id); s.ability = r.species.abilityIds['0'];
      const set = (r.species.sets || [])[0]; if (set) { /* 랜덤배틀 세트의 첫 기술들로 채움 */ }
    } catch (e) {}
    rerender();
  } });
  const head = h('div', { class: 'head' }, sprite(s.species), h('div', { class: 'col' }, spAc,
    h('div', { class: 'row' }, ...(st && st.species ? [] : []))));
  card.append(h('span', { class: 'num' }, i + 1), head);
  if (!s.species) return card;
  // 특성 · 도구 · 성격
  const abSel = h('select', { onchange: (e) => { s.ability = e.target.value; changed(); } });
  api('/api/species/' + s.species).then(r => {
    for (const [k, id] of Object.entries(r.species.abilityIds)) abSel.append(h('option', { value: id, selected: id === s.ability }, r.species.abilitiesKo[k]));
    head.querySelector('.col .row').append(...r.species.types.map(typeBadge));
  });
  const itAc = autocomplete({ kind: 'items', value: nm('items', s.item), placeholder: '도구', onPick: (it) => { s.item = it.id; changed(); renderTeam(); } });
  const natSel = h('select', { onchange: (e) => { s.nature = e.target.value; rerender(); } },
    ...S.meta.natures.map(n => h('option', { value: n.id, selected: n.id === s.nature }, n.ko + (n.plus ? ` (+${STAT_KO[n.plus]} −${STAT_KO[n.minus]})` : ' (무보정)'))));
  card.append(h('div', { class: 'grid', style: 'grid-template-columns:1fr 1fr; gap:6px' },
    h('label', { class: 'col small muted' }, '특성', abSel), h('label', { class: 'col small muted' }, '도구', itAc)),
    h('label', { class: 'col small muted' }, '성격', natSel));
  // SP
  const nat = S.meta.natures.find(n => n.id === s.nature) || {};
  const total = () => STATS.reduce((a, k) => a + (+s.sp[k] || 0), 0);
  const totEl = h('span', { class: 'small' });
  const updTot = () => { const t = total(); totEl.textContent = `SP ${t}/66`; totEl.style.color = t > 66 ? 'var(--bad)' : 'var(--muted)'; };
  const grid = h('div', { class: 'sp-grid' });
  for (const k of STATS) {
    const num = h('input', { type: 'number', min: 0, max: 32, value: s.sp[k] || 0 });
    const rng = h('input', { type: 'range', min: 0, max: 32, value: s.sp[k] || 0 });
    const set = (v) => { v = Math.max(0, Math.min(32, +v || 0)); const others = total() - (+s.sp[k] || 0); v = Math.min(v, 66 - others); s.sp[k] = v; num.value = v; rng.value = v; updTot(); changed(); };
    num.addEventListener('change', (e) => { set(e.target.value); renderTeam(); });
    rng.addEventListener('input', (e) => set(e.target.value));
    rng.addEventListener('change', () => renderTeam());
    grid.append(h('span', { class: nat.plus === k ? 'plus' : nat.minus === k ? 'minus' : '' }, STAT_KO[k]), rng, num,
      h('span', { class: 'stat' }, st && st.stats ? st.stats[k] : ''));
  }
  updTot();
  card.append(h('div', { class: 'row' }, h('b', { class: 'small' }, '능력 포인트'), totEl, st && st.mega ? h('span', { class: 'pill' }, '메가진화 후 능력치') : null), grid);
  // 기술
  const mv = h('div', { class: 'moves4' });
  for (let j = 0; j < 4; j++) {
    mv.append(autocomplete({ kind: 'moves', species: s.species, value: nm('moves', s.moves[j]), placeholder: `기술 ${j + 1}`, onPick: (it) => { s.moves[j] = it.id; s.moves = s.moves.filter(Boolean); changed(); } }));
  }
  card.append(mv, h('div', { class: 'row' },
    h('button', { class: 'btn sm', onclick: () => openBuild(i) }, 'AI 육성 추천'),
    h('button', { class: 'btn sm ghost right', onclick: () => { S.team.team[i] = emptySet(); rerender(); } }, '비우기')));
  return card;
}
$('#teamName').addEventListener('input', (e) => { S.team.name = e.target.value; saveState(); updatePill(); });
$('#newTeam').addEventListener('click', () => { S.team = { id: '', name: '', team: [] }; saveState(); renderTeam(); renderTeamList(); });
$('#saveTeam').addEventListener('click', async () => {
  const r = await api('/api/team/save', { id: S.team.id, name: S.team.name || '내 팀', team: S.team.team.filter(s => s.species) });
  S.team.id = r.id; saveState(); toast('저장했습니다'); renderTeamList();
});
$('#importBtn').addEventListener('click', () => {
  const ta = h('textarea', { placeholder: '쇼다운 형식 팀 텍스트 (영어·한글 모두 가능)\n\n예)\n한카리아스 @ 생명의구슬\n특성: 까칠한피부\n능력포인트: 2 HP / 32 공격 / 32 스피드\n성격: 명랑\n- 지진\n- 드래곤클로' });
  openModal('팀 텍스트 가져오기', ta, h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { class: 'btn primary', onclick: async () => {
    const r = await api('/api/team/parse', { text: ta.value });
    if (!r.team.length) return toast('읽은 포켓몬이 없습니다');
    S.team.team = r.team.map(t => ({ species: t.species, ability: t.ability, item: t.item, nature: t.nature, sp: t.sp, moves: t.moves, gender: t.gender }));
    saveState(); closeModal(); renderTeam(); toast(`${r.team.length}마리를 읽었습니다`);
  } }, '가져오기')));
});
$('#exportBtn').addEventListener('click', async () => {
  const ko = (await api('/api/team/export', { team: S.team.team.filter(s => s.species), lang: 'ko' })).text;
  const en = (await api('/api/team/export', { team: S.team.team.filter(s => s.species), lang: 'en' })).text;
  const ta = h('textarea', {}, ko);
  openModal('팀 텍스트 내보내기', h('div', { class: 'row', style: 'margin-bottom:8px' },
    h('button', { class: 'btn sm', onclick: () => ta.value = ko }, '한글'), h('button', { class: 'btn sm', onclick: () => ta.value = en }, '영어 (쇼다운)'),
    h('button', { class: 'btn sm primary right', onclick: () => { navigator.clipboard.writeText(ta.value); toast('복사했습니다'); } }, '복사')), ta);
});

// 육성 추천 창
async function openBuild(i) {
  const s = S.team.team[i];
  const body = h('div', { class: 'col', style: 'gap:12px' }, h('div', { class: 'empty' }, '메타 상대들과 계산 중…'));
  openModal(`${nm('species', s.species)} — AI 육성 추천`, body);
  const used = S.team.team.filter((x, k) => k !== i).map(x => x.item).filter(Boolean);
  const [sp, mv, it] = await Promise.all([api('/api/build/spread', { set: s }), api('/api/build/moves', { set: s }), api('/api/build/items', { set: s, usedItems: used })]);
  body.innerHTML = '';
  const goalKo = { offense: '공격형', defense: '방어형', balanced: '균형형' }[sp.goal] || sp.goal;
  body.append(h('div', { class: 'card' }, h('div', { class: 'hd' }, '능력 포인트·성격', h('span', { class: 'sub' }, `${goalKo}으로 판단 · ${sp.considered}가지 비교`)),
    h('table', { class: 't' }, h('tr', {}, ...['배분', '성격', '주는 데미지', '받는 데미지', '추월 비율', ''].map(x => h('th', {}, x))),
      ...sp.options.map(o => h('tr', {}, h('td', {}, o.text), h('td', {}, o.natureKo), h('td', {}, pct(o.give)), h('td', {}, pct(o.take)), h('td', {}, pct(o.outspeed)),
        h('td', {}, h('button', { class: 'btn sm', onclick: () => { s.sp = { ...o.sp }; s.nature = o.nature; saveState(); renderTeam(); toast('적용했습니다'); } }, '적용'))))),
    h('details', { style: 'padding:8px 14px' }, h('summary', { class: 'small' }, '스피드 기준표 (상대 최속 기준)'),
      h('table', { class: 't' }, h('tr', {}, h('th', {}, '상대'), h('th', {}, '최속'), h('th', {}, '추월에 필요한 내 스피드 SP')),
        ...sp.speed.map(r => h('tr', {}, h('td', {}, r.species), h('td', {}, r.theirMax), h('td', {}, r.needSp == null ? '추월 불가' : `${r.needSp}${r.needPlusNature ? ' (+스피드 성격)' : ''}`)))))));
  body.append(h('div', { class: 'card' }, h('div', { class: 'hd' }, '기술 배치', h('span', { class: 'sub' }, '메타 상대 커버 범위 기준')),
    ...mv.combos.map(c => h('div', { class: 'combo' }, h('div', { class: 'chips' }, ...c.names.map(n => h('span', { class: 'chip mv' }, n))), h('span', { class: 'muted small right' }, c.score),
      h('button', { class: 'btn sm', onclick: () => { s.moves = [...c.moves]; c.moves.forEach((m, k) => NAME.moves[m] = c.names[k]); saveState(); renderTeam(); toast('적용했습니다'); } }, '적용')))));
  body.append(h('div', { class: 'card' }, h('div', { class: 'hd' }, '도구', h('span', { class: 'sub' }, '팀 안에서 이미 쓴 도구는 아래로 (도구 중복 금지)')),
    h('table', { class: 't' }, h('tr', {}, ...['도구', '점수', '주는', '받는', '추월', ''].map(x => h('th', {}, x))),
      ...it.items.map(o => h('tr', { style: o.used ? 'opacity:.45' : '' }, h('td', {}, o.ko, o.mega ? ' (메가)' : ''), h('td', {}, o.score), h('td', {}, pct(o.give)), h('td', {}, pct(o.take)), h('td', {}, pct(o.outspeed)),
        h('td', {}, h('button', { class: 'btn sm', disabled: o.used, onclick: () => { s.item = o.item; NAME.items[o.item] = o.ko; saveState(); renderTeam(); toast('적용했습니다'); } }, '적용')))))));
}

// 모달
function openModal(title, ...nodes) { $('#modalTitle').textContent = title; const b = $('#modalBody'); b.innerHTML = ''; b.append(...nodes); $('#modal').classList.add('on'); }
function closeModal() { $('#modal').classList.remove('on'); }
$('#modalClose').addEventListener('click', closeModal);
$('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });

// ======================================================================== 선출
async function renderBring() {
  await ensureNames({ species: S.oppPreview.filter(Boolean) });
  const el = $('#oppPick'); el.innerHTML = '';
  for (let i = 0; i < 6; i++) {
    const sid = S.oppPreview[i] || '';
    el.append(h('div', { class: 'card p' }, sprite(sid, 'lg'),
      autocomplete({ kind: 'species', value: nm('species', sid), placeholder: `상대 ${i + 1}`, onPick: (it) => { S.oppPreview[i] = it.id; saveState(); renderBring(); } })));
  }
}
$('#clearOpp').addEventListener('click', () => { S.oppPreview = []; saveState(); renderBring(); });
$('#bringBtn').addEventListener('click', async () => {
  const team = S.team.team.filter(s => s.species);
  if (team.length < 4) return toast('팀 탭에서 내 팀을 먼저 채우세요');
  const opp = S.oppPreview.filter(Boolean);
  if (!opp.length) return toast('상대 포켓몬을 입력하세요');
  $('#bringRes').innerHTML = '<div class="empty">계산 중…</div>';
  const r = await api('/api/bring', { team, oppPreview: opp, gameType: S.gt });
  const lead = S.gt === 'doubles' ? 2 : 1;
  const res = $('#bringRes'); res.innerHTML = '';
  r.combos.forEach((c, k) => res.append(h('div', { class: 'combo' },
    h('b', { class: 'muted' }, k + 1), h('div', { class: 'imgs' }, ...c.order.map((i, j) => h('img', { class: j < lead ? 'lead' : '', src: '/sprite/' + team[i].species }))),
    h('div', { class: 'col', style: 'flex:1; gap:2px' }, h('b', {}, c.names.join(' · ')), h('span', { class: 'small muted' }, c.why)),
    h('span', { class: 'pill' }, c.score.toFixed(2)),
    h('button', { class: 'btn sm primary', onclick: () => startBattle(c.order, team, opp) }, '이 선출로 대전'))));
  const g = $('#bringGrid'); g.innerHTML = '';
  const tbl = h('table', { class: 't heat' }, h('tr', {}, h('th', {}), ...opp.map(s => h('th', {}, h('img', { src: '/sprite/' + s, style: 'width:36px;height:36px', title: nm('species', s) })))));
  r.grid.forEach((row, i) => tbl.append(h('tr', {}, h('td', {}, h('img', { src: '/sprite/' + team[i].species, style: 'width:36px;height:36px' })),
    ...row.map(c => { const v = c.score; const col = v > 0 ? `rgba(31,157,99,${Math.min(.8, v / 1.6)})` : `rgba(214,69,69,${Math.min(.8, -v / 1.6)})`;
      return h('td', { class: 'c', style: `background:${col}`, title: `내 ${c.move}: ${pct(c.give)} (처치 ${pct(c.giveKo)}) / 상대 ${c.oppMove}: ${pct(c.take)} (처치 ${pct(c.takeKo)})${c.faster ? ' · 내가 빠름' : ''}` }, (v > 0 ? '+' : '') + v.toFixed(1)); }))));
  g.append(tbl);
});

// ======================================================================== 대전
async function startBattle(order, team, opp) {
  const pick = S.gt === 'doubles' ? 4 : 3;
  const fullTeam = JSON.parse(JSON.stringify(S.team.team.filter(s => s.species)));
  try {  // 메가진화 가능한 포켓몬은 메가 모습도 기억해 둔다
    const st = (await api('/api/team/stats', { team: fullTeam })).stats;
    st.forEach((x, i) => { if (x.mega) fullTeam[i].megaSpecies = x.species; });
  } catch (e) {}
  const brought = order.slice(0, pick);
  const nSlots = S.gt === 'doubles' ? 2 : 1;
  S.battle = {
    id: '', gameType: S.gt, turn: 1, myTeam: fullTeam, brought, oppPreview: opp,
    mine: brought.map((idx, k) => ({ idx, hpPct: 100, status: '', boosts: {}, slot: k < nSlots ? 'ab'[k] : null, fainted: false, mega: false })),
    opp: [], field: { weather: '', terrain: '', trickRoom: false, gravity: false }, sides: { me: {}, opp: {} },
    megaUsed: { me: false, opp: false }, observations: [], turns: [], result: null, started: new Date().toISOString(),
  };
  saveState(); showPage('battle');
}
$('#newBattle').addEventListener('click', () => { if (S.battle && !confirm('지금 대전 기록을 지우고 새로 시작할까요?')) return; S.battle = null; saveState(); showPage('bring'); });

async function renderBattle() {
  const B = S.battle;
  const myA = $('#myActive'), myB = $('#myBench'), opA = $('#oppActive'), opB = $('#oppBench');
  [myA, myB, opA, opB].forEach(e => e.innerHTML = '');
  if (!B) { myA.append(h('div', { class: 'card empty' }, '선출 탭에서 대전을 시작하세요.', h('br'), h('button', { class: 'btn primary', style: 'margin-top:10px', onclick: () => showPage('bring') }, '선출하러 가기'))); return; }
  B.gameType = S.gt;
  $('#turnNo').textContent = B.turn;
  await ensureNames({ species: [...B.myTeam.map(s => s.species), ...B.oppPreview, ...B.opp.map(o => o.species)], moves: [...B.myTeam.flatMap(s => s.moves), ...B.opp.flatMap(o => o.moves || [])], items: [...B.myTeam.map(s => s.item), ...B.opp.map(o => o.item).filter(Boolean)] });
  const nSlots = B.gameType === 'doubles' ? 2 : 1;
  // 내 쪽
  for (const m of B.mine) {
    const s = B.myTeam[m.idx];
    if (m.slot && !m.fainted) myA.append(activeCard(m, s, true));
    else myB.append(benchCard(m, s, true));
  }
  // 상대: 프리뷰 중 아직 안 나온 것은 대기로
  for (const sid of B.oppPreview) if (!B.opp.find(o => o.species === sid || o.baseSpecies === sid)) B.opp.push({ species: sid, hpPct: 100, status: '', boosts: {}, slot: null, fainted: false, moves: [], item: null, ability: null, mega: false, seen: false });
  if (!B.opp.some(o => o.slot)) { B.opp.slice(0, 0); }
  for (const o of B.opp) {
    if (o.slot && !o.fainted) opA.append(activeCard(o, null, false));
    else opB.append(benchCard(o, null, false));
  }
  if (!opA.children.length) opA.append(h('div', { class: 'card empty small' }, '아래에서 상대가 낸 포켓몬을 눌러 내보내세요.'));
  // 필드
  const fw = $('#fWeather'), ft = $('#fTerrain');
  fw.innerHTML = ''; ft.innerHTML = '';
  WEATHER.forEach(([v, t]) => fw.append(h('option', { value: v, selected: B.field.weather === v }, t)));
  TERRAIN.forEach(([v, t]) => ft.append(h('option', { value: v, selected: B.field.terrain === v }, t)));
  fw.onchange = () => { B.field.weather = fw.value; changed(); };
  ft.onchange = () => { B.field.terrain = ft.value; changed(); };
  $('#fTR').checked = !!B.field.trickRoom; $('#fTR').onchange = (e) => { B.field.trickRoom = e.target.checked; changed(); };
  $('#fGrav').checked = !!B.field.gravity; $('#fGrav').onchange = (e) => { B.field.gravity = e.target.checked; changed(); };
  for (const [key, el] of [['me', '#sideMe'], ['opp', '#sideOpp']]) {
    const box = $(el); box.innerHTML = '';
    SIDE.forEach(([k, t]) => box.append(h('span', { class: 'chip' + (B.sides[key][k] ? ' on' : ''), onclick: () => {
      const cur = B.sides[key][k] || 0;
      B.sides[key][k] = k === 'spikes' ? (cur + 1) % 4 : k === 'toxicspikes' ? (cur + 1) % 3 : !cur;
      changed(); renderBattle();
    } }, t + (k === 'spikes' && B.sides[key][k] ? ' ×' + B.sides[key][k] : k === 'toxicspikes' && B.sides[key][k] ? ' ×' + B.sides[key][k] : ''))));
  }
  requestAdvice();
}
function changed() { saveState(); requestAdvice(); }

function hpClass(p) { return p > 50 ? '' : p > 20 ? 'mid' : 'low'; }
function activeCard(m, set, mine) {
  const B = S.battle;
  const sid = mine ? (m.mega ? (set.megaSpecies || set.species) : set.species) : m.species;
  const card = h('div', { class: 'card active-card' });
  const hpTxt = h('input', { type: 'number', min: 0, max: 100, value: Math.round(m.hpPct), style: 'width:64px' });
  const bar = h('div', { class: 'hpbar ' + hpClass(m.hpPct) }, h('i', { style: `width:${m.hpPct}%` }));
  const rng = h('input', { type: 'range', min: 0, max: 100, value: m.hpPct });
  const setHp = (v) => { m.hpPct = Math.max(0, Math.min(100, +v || 0)); hpTxt.value = Math.round(m.hpPct); rng.value = m.hpPct; bar.className = 'hpbar ' + hpClass(m.hpPct); bar.firstChild.style.width = m.hpPct + '%'; };
  rng.addEventListener('input', (e) => setHp(e.target.value)); rng.addEventListener('change', () => { if (m.hpPct <= 0) { m.fainted = true; m.slot = null; renderBattle(); } changed(); });
  hpTxt.addEventListener('change', (e) => { setHp(e.target.value); if (m.hpPct <= 0) { m.fainted = true; m.slot = null; renderBattle(); } changed(); });
  const title = h('div', { class: 'col' }, h('div', { class: 'row' }, h('span', { class: 'mon-name' }, nm('species', mine ? set.species : m.species) + (m.mega ? ' (메가)' : '')),
    h('span', { class: 'pill small' }, '자리 ' + (m.slot || '').toUpperCase())), bar);
  card.append(h('div', { class: 'top' }, sprite(sid, 'lg'), title));
  card.append(h('div', { class: 'hp-row' }, rng, hpTxt));
  // 상태
  card.append(h('div', { class: 'chips' }, ...STATUS.map(([v, t]) => h('span', { class: `chip st-${v}` + ((m.status || '') === v ? ' on' : ''), onclick: () => { m.status = v; changed(); renderBattle(); } }, t))));
  // 랭크
  const bx = h('div', { class: 'boosts' });
  for (const k of BOOST_KEYS) {
    const v = (m.boosts || {})[k] || 0;
    const vv = h('span', { class: 'v ' + (v > 0 ? 'pos' : v < 0 ? 'neg' : '') }, (v > 0 ? '+' : '') + v);
    const step = (d) => { m.boosts = m.boosts || {}; m.boosts[k] = Math.max(-6, Math.min(6, (m.boosts[k] || 0) + d)); changed(); renderBattle(); };
    bx.append(h('div', { class: 'boost' }, h('span', { class: 'lb' }, STAT_KO[k]), vv, h('div', { class: 'bb' }, h('button', { onclick: () => step(-1) }, '−'), h('button', { onclick: () => step(1) }, '+'))));
  }
  card.append(bx);
  // 메가 · 교체 · 쓰러짐
  const acts = h('div', { class: 'row' });
  if (mine) {
    if (set.megaSpecies && (!B.megaUsed.me || m.mega)) {
      acts.append(h('label', { class: 'chk small' }, h('input', { type: 'checkbox', checked: m.mega, onchange: (e) => { m.mega = e.target.checked; if (m.mega) B.megaUsed.me = true; changed(); renderBattle(); } }), '메가진화함'));
    }
  } else {
    acts.append(h('label', { class: 'chk small' }, h('input', { type: 'checkbox', checked: m.mega, onchange: async (e) => {
      if (e.target.checked) {
        const r = await api('/api/species/' + m.species);
        const megas = r.species.megas || [];
        if (!megas.length) { toast('메가진화할 수 없는 포켓몬입니다'); e.target.checked = false; return; }
        let pickM = megas[0];
        if (megas.length > 1) { const k = prompt(megas.map((x, i) => `${i + 1}. ${x.ko}`).join('\n') + '\n번호를 고르세요', '1'); pickM = megas[(+k || 1) - 1] || megas[0]; }
        m.baseSpecies = m.species; m.species = pickM.id; m.item = pickM.item || m.item; m.mega = true; B.megaUsed.opp = true; NAME.species[pickM.id] = pickM.ko;
      } else { if (m.baseSpecies) m.species = m.baseSpecies; m.mega = false; }
      changed(); renderBattle(); } }), '메가진화함'));
  }
  acts.append(h('button', { class: 'btn sm ghost right', onclick: () => { m.slot = null; changed(); renderBattle(); } }, '들여보내기'),
    h('button', { class: 'btn sm ghost', onclick: () => { m.fainted = true; m.hpPct = 0; m.slot = null; changed(); renderBattle(); } }, '쓰러짐'));
  card.append(acts);
  if (!mine) {
    // 드러난 기술 · 도구 · 특성
    const mvs = h('div', { class: 'chips' }, h('span', { class: 'small muted' }, '쓴 기술'),
      ...(m.moves || []).map(id => h('span', { class: 'chip mv' }, nm('moves', id), h('span', { class: 'x', onclick: () => { m.moves = m.moves.filter(x => x !== id); changed(); renderBattle(); } }, '×'))));
    const add = autocomplete({ kind: 'moves', species: m.species, placeholder: '+ 상대가 쓴 기술', onPick: (it) => { if (!m.moves.includes(it.id)) m.moves.push(it.id); changed(); renderBattle(); } });
    add.style.width = '170px';
    mvs.append(add);
    card.append(mvs);
    const itemAc = autocomplete({ kind: 'items', value: m.item ? nm('items', m.item) : '', placeholder: m.item === '' ? '없음(소모·탈취)' : '모름', onPick: (it) => { m.item = it.id; changed(); } });
    card.append(h('div', { class: 'grid', style: 'grid-template-columns:1fr auto auto; gap:6px; align-items:center' }, itemAc,
      h('button', { class: 'btn sm', onclick: () => { m.item = null; changed(); renderBattle(); } }, '모름'),
      h('button', { class: 'btn sm', onclick: () => { m.item = ''; changed(); renderBattle(); } }, '없음')));
  } else {
    card.append(h('div', { class: 'chips' }, ...set.moves.map(id => h('span', { class: 'chip mv' }, nm('moves', id)))));
  }
  return card;
}
function benchCard(m, set, mine) {
  const B = S.battle;
  const sid = mine ? set.species : m.species;
  const nSlots = B.gameType === 'doubles' ? 2 : 1;
  const list = mine ? B.mine : B.opp;
  const free = () => { const used = list.filter(x => x.slot && !x.fainted).map(x => x.slot); return ['a', 'b'].slice(0, nSlots).find(s => !used.includes(s)); };
  return h('div', { class: 'card b' + (m.fainted ? ' fnt' : ''), title: m.fainted ? '쓰러짐 — 눌러서 되살리기' : '눌러서 내보내기', onclick: () => {
    if (m.fainted) { if (confirm('쓰러짐을 취소할까요?')) { m.fainted = false; m.hpPct = Math.max(m.hpPct, 1); changed(); renderBattle(); } return; }
    let slot = free();
    if (!slot) { // 자리가 차 있으면 첫 자리와 교체
      const cur = list.find(x => x.slot === 'a' && !x.fainted); if (cur) cur.slot = null; slot = 'a';
    }
    m.slot = slot; if (!mine) m.seen = true; changed(); renderBattle();
  } }, sprite(sid, 'sm'), h('div', { class: 'col' }, h('span', { class: 'nm' }, nm('species', sid)),
    h('div', { class: 'hpbar ' + hpClass(m.hpPct) }, h('i', { style: `width:${m.hpPct}%` })),
    h('span', { class: 'small muted' }, m.fainted ? '쓰러짐' : `${Math.round(m.hpPct)}%` + (m.status ? ' · ' + STATUS.find(s => s[0] === m.status)[1] : '') + (!mine && !m.seen ? ' · 미등장' : ''))));
}
$('#turnUp').addEventListener('click', () => { if (!S.battle) return; S.battle.turn++; changed(); $('#turnNo').textContent = S.battle.turn; });
$('#turnDown').addEventListener('click', () => { if (!S.battle) return; S.battle.turn = Math.max(1, S.battle.turn - 1); changed(); $('#turnNo').textContent = S.battle.turn; });

// 추천 요청
const requestAdvice = debounce(async () => {
  const B = S.battle;
  if (!B) return;
  const nSlots = B.gameType === 'doubles' ? 2 : 1;
  const myActive = B.mine.filter(m => m.slot && !m.fainted).length;
  const state = { ...B, forceSwitch: myActive < nSlots && B.mine.some(m => !m.slot && !m.fainted), useSearch: $('#useSearch').checked };
  $('#adviceSrc').textContent = '계산 중…';
  try {
    const r = await api('/api/advise', state);
    S.advice = r;
    renderAdvice();
  } catch (e) { $('#adviceSrc').textContent = '오류: ' + e.message; }
}, 250);
$('#adviseBtn').addEventListener('click', () => requestAdvice());
$('#useSearch').checked = store.get('useSearch', false);
$('#useSearch').addEventListener('change', (e) => { store.set('useSearch', e.target.checked); requestAdvice(); });

function renderAdvice() {
  const r = S.advice;
  $('#adviceSrc').textContent = r.source;
  const v = r.winNow;
  $('#gaugeBar').style.width = (v == null ? 50 : v * 100) + '%';
  $('#gaugeTxt').textContent = v == null ? '–' : Math.round(v * 100) + '%';
  $('#advCount').textContent = `선택지 ${r.nOptions}개`;
  const el = $('#advList'); el.innerHTML = '';
  if (!r.options.length) { el.append(h('div', { class: 'empty' }, '고를 수 있는 행동이 없습니다. 내 포켓몬을 내보내세요.')); }
  const hasWin = r.options.some(o => o.win != null);
  const hasProb = r.options.some(o => o.prob != null);
  const maxScore = Math.max(...r.options.map(o => o.score), 0.01);
  r.options.slice(0, 12).forEach((o, i) => {
    const val = hasWin ? o.win : hasProb ? o.prob : Math.max(0, o.score / maxScore);
    const txt = hasWin ? Math.round(o.win * 100) + '%' : hasProb ? Math.round(o.prob * 100) + '%' : o.score.toFixed(2);
    const sub = hasWin ? '예상 승률' : hasProb ? '추천 확률' : '점수';
    el.append(h('div', { class: 'opt' }, h('span', { class: 'rank' }, i + 1),
      h('div', {}, h('div', { class: 'lbl' }, o.label), o.why ? h('div', { class: 'why' }, o.why) : null, h('div', { class: 'bar' }, h('i', { style: `width:${Math.round(val * 100)}%` }))),
      h('span', { class: 'pct', title: sub }, txt, h('div', { class: 'small muted', style: 'font-weight:400;font-size:10px' }, sub))));
  });
  renderDetail();
}
let detailTab = 'dmg';
$$('#detailTabs button').forEach(b => b.addEventListener('click', () => { detailTab = b.dataset.d; $$('#detailTabs button').forEach(x => x.classList.toggle('on', x === b)); renderDetail(); }));
function koCell(p, n) { return p >= 0.999 ? h('span', { class: 'ko1' }, `확정 ${n}타`) : p > 0 ? h('span', { class: n === 1 ? 'ko1' : 'ko2' }, `${n}타 ${Math.round(p * 100)}%`) : ''; }
function renderDetail() {
  const r = S.advice; const el = $('#detail'); el.innerHTML = '';
  if (!r) return;
  if (detailTab === 'dmg') {
    el.append(h('table', { class: 't' }, h('tr', {}, ...['기술', '대상', '데미지', '처치'].map(x => h('th', {}, x))),
      ...r.damage.map(d => h('tr', {}, h('td', {}, typeBadge(d.type), ' ', d.move), h('td', {}, d.to), h('td', {}, d.immune ? '효과 없음' : `${d.min}~${d.max}%`), h('td', {}, d.immune ? '' : (koCell(d.ko1, 1) || koCell(d.ko2, 2)))))));
  } else if (detailTab === 'threat') {
    el.append(h('div', { class: 'small muted', style: 'padding:8px 12px' }, '상대가 가졌을 법한 기술 (확률은 세트 추정 · 드러난 기술은 100%)'),
      h('table', { class: 't' }, h('tr', {}, ...['상대 기술', '가능성', '대상', '데미지', '처치'].map(x => h('th', {}, x))),
        ...r.threats.map(d => h('tr', {}, h('td', {}, d.from, ' · ', d.move, d.revealed ? ' ✔' : ''), h('td', {}, pct(d.prob)), h('td', {}, d.to), h('td', {}, d.immune ? '효과 없음' : `${d.min}~${d.max}%`), h('td', {}, koCell(d.ko1, 1))))));
  } else if (detailTab === 'speed') {
    el.append(h('div', { style: 'padding:10px 12px' }, r.speed.trickRoom ? h('div', { class: 'pill' }, '트릭룸: 느린 쪽이 먼저') : null,
      h('ol', {}, ...r.speed.order.map(o => h('li', {}, h('b', { style: `color:var(${o.mine ? '--mine' : '--opp'})` }, o.who), ` ${o.speed}`, o.estimated ? h('span', { class: 'muted small' }, ' (추정)') : '')))));
  } else {
    for (const [sid, b] of Object.entries(r.oppSets)) {
      const nar = r.narrowed[sid];
      el.append(h('div', { style: 'padding:10px 12px; border-bottom:1px solid var(--line)' }, h('b', {}, nm('species', sid)), ' ', h('span', { class: 'muted small' }, `${b.role || ''} · 특성 추정: ${b.ability || '?'}`),
        h('div', { class: 'chips', style: 'margin-top:6px' }, ...b.moves.map(m => h('span', { class: 'chip mv', title: Math.round(m.prob * 100) + '%' }, `${m.move} ${Math.round(m.prob * 100)}%`))),
        nar ? h('div', { class: 'small', style: 'margin-top:6px' }, `관측 데미지와 맞는 조합 ${nar.consistent}/${nar.total}개` + (nar.best ? ` · 가장 유력: ${Object.entries(nar.best.sp).map(([k, v]) => v + ' ' + STAT_KO[k]).join(' / ') || '무투자'} ${nar.best.nature} ${nar.best.item || '도구 없음'}` : '')) : null));
    }
  }
}

// 관측 데미지
$('#obsBtn').addEventListener('click', () => {
  const B = S.battle; if (!B) return toast('대전을 먼저 시작하세요');
  const opps = B.opp.filter(o => o.seen || o.slot);
  if (!opps.length) return toast('상대 포켓몬을 먼저 내보내세요');
  const dir = selectEl([['opp2me', '상대 → 내 포켓몬'], ['me2opp', '내 포켓몬 → 상대']], 'opp2me', () => {});
  const opp = selectEl(opps.map(o => [o.species, nm('species', o.species)]), (opps.find(o => o.slot) || opps[0]).species, () => {});
  const mine = selectEl(B.mine.map(m => [String(m.idx), nm('species', B.myTeam[m.idx].species)]), String((B.mine.find(m => m.slot) || B.mine[0]).idx), () => {});
  let move = '';
  const mv = autocomplete({ kind: 'moves', placeholder: '기술', onPick: (it) => move = it.id });
  const p = h('input', { type: 'number', min: 0, max: 100, placeholder: '깎인 %', style: 'width:90px' });
  const crit = h('input', { type: 'checkbox' });
  const list = h('div', { class: 'col small' }, ...(B.observations || []).map((o, i) => h('div', { class: 'row' }, `${o.dir === 'opp2me' ? '상대→나' : '나→상대'} ${nm('species', o.oppSpecies)} · ${nm('moves', o.move)} · ${o.pct}%${o.crit ? ' (급소)' : ''}`,
    h('button', { class: 'btn sm ghost', onclick: () => { B.observations.splice(i, 1); changed(); closeModal(); } }, '삭제'))));
  openModal('관측 데미지 입력', h('div', { class: 'col', style: 'gap:10px' },
    h('div', { class: 'small muted' }, '실제로 깎인 HP(%)를 넣으면, 그 데미지가 나올 수 있는 상대의 SP·성격·도구 조합만 남겨 계산에 반영합니다. 상대 HP는 체력바 기준이라 ±2% 오차를 허용합니다.'),
    h('div', { class: 'grid g2' }, h('label', { class: 'col small' }, '방향', dir), h('label', { class: 'col small' }, '상대 포켓몬', opp),
      h('label', { class: 'col small' }, '내 포켓몬', mine), h('label', { class: 'col small' }, '기술', mv)),
    h('div', { class: 'row' }, p, h('label', { class: 'chk small' }, crit, '급소였음'),
      h('button', { class: 'btn primary right', onclick: () => {
        if (!move || p.value === '') return toast('기술과 % 를 넣으세요');
        const o = B.opp.find(x => x.species === opp.value);
        B.observations = B.observations || [];
        B.observations.push({ dir: dir.value, oppSpecies: opp.value, move, pct: +p.value, my_idx: +mine.value, crit: crit.checked, oppHpPctBefore: dir.value === 'me2opp' ? null : null });
        if (dir.value === 'opp2me' && o && !o.moves.includes(move)) o.moves.push(move);
        changed(); closeModal(); renderBattle(); toast('관측을 반영했습니다');
      } }, '추가')),
    h('b', { class: 'small' }, '지금까지의 관측'), list));
});

// 턴 기록 · 결과
$('#nextTurn').addEventListener('click', async () => {
  const B = S.battle; if (!B) return;
  const top = S.advice && S.advice.options[0];
  B.turns.push({ turn: B.turn, winNow: S.advice ? S.advice.winNow : null, top: top ? top.label : null,
    state: JSON.parse(JSON.stringify({ mine: B.mine, opp: B.opp, field: B.field, sides: B.sides })) });
  B.turn++; saveState(); await saveBattle(); renderBattle(); toast(`${B.turn - 1}턴을 기록했습니다`);
});
async function saveBattle() { const B = S.battle; const r = await api('/api/battle/save', B); B.id = r.id; saveState(); }
async function finish(res) {
  const B = S.battle; if (!B) return;
  B.result = res; B.turns.push({ turn: B.turn, winNow: S.advice ? S.advice.winNow : null, final: true });
  await saveBattle(); toast(res === 'win' ? '승리 기록! 학습 데이터로 저장했습니다' : '패배 기록. 분석에 쓰겠습니다');
}
$('#winBtn').addEventListener('click', () => finish('win'));
$('#loseBtn').addEventListener('click', () => finish('lose'));

// ======================================================================== 계산기
const CALC = store.get('calc', { a: emptySet(), d: emptySet(), aState: { hpPct: 100, status: '', boosts: {} }, dState: { hpPct: 100, status: '', boosts: {} }, field: {} });
function calcSide(key) {
  const s = CALC[key], st = CALC[key + 'State'];
  const box = h('div', { class: 'col', style: 'gap:8px' });
  const fromTeam = h('select', { onchange: (e) => { if (e.target.value === '') return; CALC[key] = JSON.parse(JSON.stringify(S.team.team[+e.target.value])); store.set('calc', CALC); renderCalc(); } },
    h('option', { value: '' }, '내 팀에서 불러오기…'), ...S.team.team.map((t, i) => t.species ? h('option', { value: i }, nm('species', t.species)) : null));
  box.append(h('div', { class: 'row' }, sprite(s.species), autocomplete({ kind: 'species', value: nm('species', s.species), placeholder: '포켓몬', onPick: async (it) => {
    s.species = it.id; const r = await api('/api/species/' + it.id); s.ability = r.species.abilityIds['0']; s.moves = []; store.set('calc', CALC); renderCalc(); } }), fromTeam));
  if (!s.species) return box;
  box.append(h('div', { class: 'grid g2' },
    h('label', { class: 'col small muted' }, '도구', autocomplete({ kind: 'items', value: nm('items', s.item), onPick: (it) => { s.item = it.id; store.set('calc', CALC); } })),
    h('label', { class: 'col small muted' }, '성격', selectEl(S.meta.natures.map(n => [n.id, n.ko]), s.nature, (v) => { s.nature = v; store.set('calc', CALC); }))));
  box.append(h('label', { class: 'col small muted' }, '특성', autocomplete({ kind: 'abilities', species: s.species, value: nm('abilities', s.ability), onPick: (it) => { s.ability = it.id; store.set('calc', CALC); } })));
  box.append(h('div', { class: 'row small' }, ...STATS.map(k => h('label', { class: 'col' }, STAT_KO[k], h('input', { type: 'number', min: 0, max: 32, value: (s.sp || {})[k] || 0, style: 'width:56px', onchange: (e) => { s.sp = s.sp || {}; s.sp[k] = +e.target.value; store.set('calc', CALC); } })))));
  box.append(h('div', { class: 'row small' }, 'HP %', h('input', { type: 'number', value: st.hpPct, style: 'width:70px', onchange: (e) => { st.hpPct = +e.target.value; store.set('calc', CALC); } }),
    selectEl(STATUS, st.status, (v) => { st.status = v; store.set('calc', CALC); }),
    ...BOOST_KEYS.map(k => h('label', { class: 'col' }, STAT_KO[k], h('input', { type: 'number', min: -6, max: 6, value: st.boosts[k] || 0, style: 'width:50px', onchange: (e) => { st.boosts[k] = +e.target.value; store.set('calc', CALC); } })))));
  if (key === 'a') {
    const mv = h('div', { class: 'moves4' });
    for (let j = 0; j < 4; j++) mv.append(autocomplete({ kind: 'moves', species: s.species, value: nm('moves', (s.moves || [])[j]), placeholder: '기술 ' + (j + 1), onPick: (it) => { s.moves = s.moves || []; s.moves[j] = it.id; store.set('calc', CALC); } }));
    box.append(mv);
  }
  return box;
}
function renderCalc() {
  ensureNames({ species: [CALC.a.species, CALC.d.species], moves: CALC.a.moves || [], items: [CALC.a.item, CALC.d.item], abilities: [CALC.a.ability, CALC.d.ability] }).then(() => {
    $('#calcA').innerHTML = ''; $('#calcA').append(calcSide('a'));
    $('#calcD').innerHTML = ''; $('#calcD').append(calcSide('d'));
    const f = CALC.field; const fe = $('#calcField'); fe.innerHTML = '';
    fe.append(selectEl(WEATHER, f.weather || '', (v) => { f.weather = v; store.set('calc', CALC); }), selectEl(TERRAIN, f.terrain || '', (v) => { f.terrain = v; store.set('calc', CALC); }),
      ...[['reflect', '리플렉터'], ['lightscreen', '빛의장막'], ['helpingHand', '도우미'], ['spread', '범위기(더블)']].map(([k, t]) =>
        h('label', { class: 'chk small' }, h('input', { type: 'checkbox', checked: !!f[k], onchange: (e) => { f[k] = e.target.checked; store.set('calc', CALC); } }), t)));
  });
}
$('#calcBtn').addEventListener('click', async () => {
  const f = CALC.field;
  try {
    const r = await api('/api/calc', { attacker: { ...CALC.a, ...CALC.aState }, defender: { ...CALC.d, ...CALC.dState },
      field: { weather: f.weather, terrain: f.terrain, helpingHand: f.helpingHand, spread: f.spread || null, gameType: f.spread ? 'doubles' : 'singles', defSide: { reflect: f.reflect, lightscreen: f.lightscreen } } });
    const el = $('#calcRes'); el.innerHTML = '';
    el.append(h('div', { class: 'small muted', style: 'padding:8px 14px' }, `상대 HP ${r.defender.hp}/${r.defender.maxhp} · 실능력치 ${STATS.map(k => STAT_KO[k] + ' ' + (k === 'hp' ? r.defender.maxhp : r.defender.stats[k])).join(' / ')}`),
      h('table', { class: 't' }, h('tr', {}, ...['기술', '위력', '데미지 %', '급소', '처치 확률 (1·2·3타)', '16단계 난수'].map(x => h('th', {}, x))),
        ...r.results.map(x => h('tr', {}, h('td', {}, typeBadge(x.type), ' ', x.move), h('td', {}, x.bp || '-'),
          h('td', {}, x.immune ? '효과 없음' : x.blocked ? `막힘(${x.blocked})` : `${x.min}~${x.max}%`), h('td', {}, x.immune ? '' : `${x.critMin}~${x.critMax}%`),
          h('td', {}, x.ko.slice(0, 3).map(p => pct(p)).join(' · ')), h('td', { class: 'small muted' }, (x.rolls || []).join(' '), x.notes.length ? h('div', { style: 'color:var(--warn)' }, x.notes.join(' / ')) : null)))));
  } catch (e) { toast(e.message); }
});

// ======================================================================== AI·학습
async function renderAI() {
  const [info, meta, rep] = await Promise.all([api('/api/ai/info'), api('/api/meta'), api('/api/battle/report')]);
  const tiles = $('#aiTiles'); tiles.innerHTML = '';
  for (const gt of ['singles', 'doubles']) {
    const m = meta.models[gt]; const mt = (m && m.metrics) || {};
    tiles.append(h('div', { class: 'card tile' }, h('div', { class: 'k' }, (gt === 'singles' ? '싱글' : '더블') + ' 신경망'), h('div', { class: 'v' }, m ? (m.games ? `${m.games}판` : '학습됨') : '없음'),
      h('div', { class: 'small muted' }, m ? [mt.top1 != null && `행동 일치 ${pct(mt.top1)}`, mt.valueAcc != null && `승패 예측 ${pct(mt.valueAcc)}`, `입력 ${m.inputs}칸`, (m.created || '').slice(0, 10)].filter(Boolean).join(' · ') : '학습 전 — 선생님 AI 로 추천')));
  }
  const dd = info.data || {};
  tiles.append(h('div', { class: 'card tile' }, h('div', { class: 'k' }, '사람 기보'), h('div', { class: 'v' }, `${dd.replays || 0}판`),
    h('div', { class: 'small muted' }, `쇼다운 리플레이 · 나와 둔 대전 ${dd.vsUser || 0}판`)));
  if (dd.patch) tiles.append(h('div', { class: 'card tile' }, h('div', { class: 'k' }, '패치 감지'), h('div', { class: 'v' }, dd.patch.changes ? `${dd.patch.changes}건 변경` : '변경 없음'),
    h('div', { class: 'small muted' }, dd.patch.changes ? '집중 대전: ' + dd.patch.focus.slice(0, 5).map(([k]) => k).join(', ') : '지난 학습 이후 데이터 같음')));
  tiles.append(h('div', { class: 'card tile' }, h('div', { class: 'k' }, '내 실제 대전'), h('div', { class: 'v' }, `${rep.wins}/${rep.total}`), h('div', { class: 'small muted' }, rep.total ? `승률 ${pct(rep.wins / rep.total)}` : '기록 없음')));
  const st = info.selftest || {};
  tiles.append(h('div', { class: 'card tile' }, h('div', { class: 'k' }, '마지막 셀프테스트'), h('div', { class: 'v', style: `color:var(${st.ok ? '--good' : '--bad'})` }, st.checks ? (st.ok ? '통과' : '실패') : '–'),
    h('div', { class: 'small muted' }, st.checks && st.checks.damage ? st.checks.damage.summary : '')));
  // 리그
  const lg = $('#aiLeague'); lg.innerHTML = '';
  for (const gt of ['singles', 'doubles']) {
    const m = meta.models[gt]; if (!m || !m.league) continue;
    lg.append(h('b', {}, gt === 'singles' ? '싱글' : '더블'), h('table', { class: 't', style: 'margin:6px 0 14px' }, h('tr', {}, h('th', {}, '상대'), h('th', {}, '전적'), h('th', {}, '승률')),
      ...Object.values(m.league).map(r => h('tr', {}, h('td', {}, r.ko), h('td', {}, `${r.wins}/${r.played}`), h('td', {}, h('div', { class: 'gauge' }, h('div', { class: 'g' }, h('i', { style: `width:${r.winrate * 100}%` })), h('b', { style: 'font-size:13px' }, pct(r.winrate))))))));
  }
  if (!lg.children.length) lg.append(h('div', { class: 'empty' }, '아직 리그 평가 기록이 없습니다.'));
  renderSelftest(st);
  // 장부
  const led = info.ledger || {}; const le = $('#aiLedger'); le.innerHTML = '';
  const stKo = { adopted: '채택', candidate: '후보', rejected: '제외' };
  le.append(h('table', { class: 't' }, h('tr', {}, ...['묶음', '설명', '종류', '칸', '추가일', '상태', '빼면 손실 증가', '실험 데이터'].map(x => h('th', {}, x))),
    ...Object.entries(led.groups || {}).map(([k, e]) => { const t = (e.tests || []).slice(-1)[0];
      return h('tr', {}, h('td', {}, h('b', {}, e.ko), h('div', { class: 'small muted' }, k)), h('td', { class: 'small' }, e.desc), h('td', {}, e.kind === 'state' ? '상태' : '행동'), h('td', {}, e.size), h('td', {}, e.added),
        h('td', { class: 'st-' + e.status }, stKo[e.status] || e.status), h('td', {}, t ? (t.delta > 0 ? '+' : '') + t.delta.toFixed(4) : '-'), h('td', { class: 'small muted' }, t ? `${t.games}판` : '-')); })));
  if ((led.suggestions || []).length) {
    le.append(h('h4', { style: 'margin:14px 0 6px' }, '다음에 넣을 변수 후보 (대전 후 분석)'), h('ul', {}, ...led.suggestions.slice(-20).map(s => h('li', {}, h('b', {}, s.name), ' — ', s.why, h('span', { class: 'muted small' }, ` (${s.evidence})`)))));
  }
  // 기록
  const hi = $('#aiHistory'); hi.innerHTML = '';
  const ov = (info.overnight || []).slice().reverse();
  if (!ov.length) hi.append(h('div', { class: 'empty' }, '밤샘 학습 기록이 없습니다. 밤샘학습.bat 으로 시작하세요.'));
  else hi.append(h('table', { class: 't' }, h('tr', {}, ...['회차', '시각', '형식', '판수', '쇼다운 AI', '선생님', '이전 최고', '교체'].map(x => h('th', {}, x))),
    ...ov.map(e => { const L = e.league || {}; const w = (k) => L[k] ? pct(L[k].winrate) : '-';
      return h('tr', {}, h('td', {}, e.iter), h('td', {}, (e.when || '').slice(5, 16).replace('T', ' ')), h('td', {}, e.gameType === 'singles' ? '싱글' : '더블'), h('td', {}, e.games),
        h('td', {}, w('showdown_random')), h('td', {}, w('heuristic')), h('td', {}, w('best')), h('td', {}, e.kept ? '예' : '아니오')); })));
  // 내 대전
  const re = $('#aiReport'); re.innerHTML = '';
  if (!rep.total) re.append(h('div', { class: 'empty' }, '대전 탭에서 승리/패배를 기록하면 여기서 분석합니다.'));
  else {
    re.append(h('table', { class: 't' }, h('tr', {}, h('th', {}, '예측 승률 구간'), h('th', {}, '턴 수'), h('th', {}, '실제 승률')),
      ...rep.calibration.map(c => h('tr', {}, h('td', {}, c.bin), h('td', {}, c.n), h('td', {}, c.actual == null ? '-' : pct(c.actual))))),
      h('h4', { style: 'margin:12px 0 6px' }, '자주 만난 상대'), h('div', { class: 'chips' }, ...rep.topOpp.map(o => h('span', { class: 'chip mv' }, `${o.ko} ×${o.n}`))));
  }
}
function renderSelftest(st) {
  const el = $('#aiSelftest'); el.innerHTML = '';
  if (!st || !st.checks) { el.append(h('div', { class: 'empty' }, '아직 실행하지 않았습니다.')); return; }
  const names = { data: '데이터', stats: '능력치 계산', damage: '데미지 (쇼다운 대조)', paste: '팀 텍스트', features: '변수 등록부', sim: '자기대전', model: '학습 모델' };
  el.append(h('table', { class: 't' }, ...Object.entries(st.checks).map(([k, r]) => h('tr', {}, h('td', {}, h('b', { style: `color:var(${r.ok ? '--good' : '--bad'})` }, r.ok ? '통과' : '실패')), h('td', {}, names[k] || k), h('td', { class: 'small' }, r.summary || r.skipped || (r.errors || []).join(' / '))))));
}
$('#selftestBtn').addEventListener('click', async () => {
  await api('/api/selftest', {}); $('#aiSelftest').innerHTML = '<div class="empty">셀프테스트 실행 중… (데미지 대조 수천 건)</div>';
  const poll = async () => { const r = await api('/api/selftest/status'); if (r.running) return setTimeout(poll, 800); renderSelftest(r.result); toast(r.result && r.result.ok ? '셀프테스트 통과' : '셀프테스트 실패 항목이 있습니다'); };
  setTimeout(poll, 800);
});

// ======================================================================== 작업 기다리기
async function waitJob(jid, onTick) {
  for (;;) {
    const r = await api('/api/job/' + jid);
    onTick && onTick(r.job);
    if (!r.job.running) { if (r.job.error) throw new Error(r.job.error); return r.job.result; }
    await new Promise(res => setTimeout(res, 1200));
  }
}
async function simReady(el) {
  const r = await api('/api/sim/status');
  if (el) el.textContent = r.available ? '시뮬레이션 준비됨 (쇼다운 엔진)' : r.hint;
  return r.available;
}

// ======================================================================== 파티 짜기
let BOXS = { species: [], sets: {} };
async function renderBuild() {
  try { BOXS = await api('/api/box'); } catch (e) {}
  await ensureNames({ species: BOXS.species });
  const add = $('#boxAdd'); add.innerHTML = '';
  add.append(autocomplete({ kind: 'species', placeholder: '+ 가진 포켓몬 추가', noMega: true, onPick: (it) => {
    if (!BOXS.species.includes(it.id)) BOXS.species.push(it.id); saveBox(); renderBuild(); } }));
  const list = $('#boxList'); list.innerHTML = '';
  BOXS.species.forEach(sid => list.append(h('span', { class: 'chip mv', style: 'display:inline-flex;align-items:center;gap:4px' },
    h('img', { src: '/sprite/' + sid, style: 'width:24px;height:24px' }), nm('species', sid),
    h('span', { class: 'x', onclick: () => { BOXS.species = BOXS.species.filter(x => x !== sid); saveBox(); renderBuild(); } }, '×'))));
  $('#boxCount').textContent = BOXS.species.length ? `${BOXS.species.length}마리` : '비어 있음';
  simReady($('#simHint'));
}
const saveBox = () => api('/api/box', BOXS).catch(() => {});
$('#boxClear').addEventListener('click', () => { if (!confirm('보관함을 비울까요?')) return; BOXS.species = []; saveBox(); renderBuild(); });
$('#boxPaste').addEventListener('click', () => {
  const ta = h('textarea', { placeholder: '포켓몬 이름을 줄바꿈이나 쉼표로 구분해 넣으세요 (한글·영어)\n예) 한카리아스, 리자몽, 팬텀\n또는 쇼다운 팀 텍스트를 그대로 붙여넣어도 됩니다' });
  openModal('보관함에 여러 마리 넣기', ta, h('button', { class: 'btn primary', style: 'margin-top:10px', onclick: async () => {
    const text = ta.value;
    let ids = [];
    if (text.includes('@') || text.includes('- ')) ids = (await api('/api/team/parse', { text })).team.map(t => t.species);
    else for (const name of text.split(/[\n,]+/).map(x => x.trim()).filter(Boolean)) {
      const r = await api('/api/search?' + new URLSearchParams({ kind: 'species', q: name, limit: 1 }));
      if (r.results[0]) { ids.push(r.results[0].id); NAME.species[r.results[0].id] = r.results[0].ko; }
    }
    for (const i of ids) if (i && !BOXS.species.includes(i)) BOXS.species.push(i);
    saveBox(); closeModal(); renderBuild(); toast(`${ids.length}마리를 넣었습니다`);
  } }, '넣기'));
});
function teamCards(team) {
  return h('div', { class: 'grid g3' }, ...team.map(s => h('div', { class: 'card', style: 'padding:10px' },
    h('div', { class: 'row' }, h('img', { src: '/sprite/' + s.species, style: 'width:48px;height:48px' }), h('div', { class: 'col', style: 'gap:2px' },
      h('b', {}, s.ko), h('span', { class: 'small muted' }, `${s.abilityKo} · ${s.itemKo || '도구 없음'}`))),
    h('div', { class: 'small' }, `${s.natureKo} · ${Object.entries(s.sp || {}).map(([k, v]) => v + ' ' + STAT_KO[k]).join(' / ')}`),
    h('div', { class: 'chips', style: 'margin-top:4px' }, ...s.movesKo.map(m => h('span', { class: 'chip mv' }, m))))));
}
function applyTeam(team) {
  S.team = { id: '', name: 'AI 추천 파티', team: team.map(s => ({ species: s.species, ability: s.ability, item: s.item, nature: s.nature, sp: { ...s.sp }, moves: [...s.moves] })) };
  team.forEach(s => { NAME.species[s.species] = s.ko; s.moves.forEach((m, i) => NAME.moves[m] = s.movesKo[i]); NAME.items[s.item] = s.itemKo; NAME.abilities[s.ability] = s.abilityKo; });
  saveState(); renderTeam(); toast('팀 탭에 넣었습니다 — 저장을 눌러 보관하세요');
}
$('#buildBtn').addEventListener('click', async () => {
  const res = $('#buildRes');
  const locked = $('#buildLock').checked ? S.team.team.filter(s => s.species) : [];
  try {
    const { job } = await api('/api/team/build', { gameType: S.gt, box: BOXS.species, locked, steps: +$('#buildSteps').value || 20 });
    res.innerHTML = '<div class="empty">파티를 평가하는 중…</div>';
    const r = await waitJob(job, (j) => {
      const st = j.step ? ` · 다듬기 ${j.step[0]}/${j.step[1]} · 지금 승률 ${Math.round(j.step[2] * 100)}%` : '';
      $('#buildStatus').textContent = (j.progress.slice(-1)[0] || '') + st;
      if (j.partial && !res.dataset.partial) { res.innerHTML = ''; res.append(h('div', { class: 'small muted' }, '출발 파티'), teamCards(j.partial)); res.dataset.partial = '1'; }
    });
    delete res.dataset.partial;
    res.innerHTML = '';
    res.append(h('div', { class: 'row', style: 'margin-bottom:10px' }, h('b', { style: 'font-size:18px' }, `메타 상대 예상 승률 ${Math.round(r.score * 100)}%`),
      h('span', { class: 'small muted' }, `평가 ${r.evals}번 · 상대 파티 ${r.opponents.length}개`), h('button', { class: 'btn primary right', onclick: () => applyTeam(r.team) }, '이 파티를 팀 탭에 넣기')),
      teamCards(r.team),
      h('h4', { style: 'margin:14px 0 6px' }, '승률을 올린 변경'), h('ul', { class: 'small' }, ...(r.history.length > 1 ? r.history.slice(1).map(x => h('li', {}, `${x.change} → ${Math.round(x.score * 100)}%`)) : [h('li', {}, '출발 파티보다 나은 변경을 찾지 못함 (다듬기 횟수를 늘려 보세요)')])),
      h('h4', { style: 'margin:14px 0 6px' }, '상대 파티별 승률'), h('table', { class: 't' }, ...r.opponents.map((o, i) => h('tr', {}, h('td', { class: 'small' }, o.join(' · ')), h('td', {}, Math.round(r.perOpp[i] * 100) + '%')))));
  } catch (e) { res.innerHTML = ''; res.append(h('div', { class: 'empty' }, e.message)); }
});
$('#oneBtn').addEventListener('click', async () => {
  const res = $('#buildRes');
  const team = S.team.team.filter(s => s.species);
  if (team.length >= 6) return toast('팀 탭의 파티가 이미 6마리입니다. 한 마리를 비우고 눌러 주세요.');
  try {
    const { job } = await api('/api/team/one', { gameType: S.gt, box: BOXS.species, team });
    res.innerHTML = '<div class="empty">후보를 하나씩 넣어 대전시키는 중…</div>';
    const r = await waitJob(job, (j) => { $('#buildStatus').textContent = j.progress.slice(-1)[0] || ''; });
    res.innerHTML = '';
    res.append(h('div', { class: 'small muted', style: 'margin-bottom:8px' }, `지금 팀(${team.length}마리)에 넣었을 때 메타 상대 승률`),
      ...r.rows.map((x, i) => h('div', { class: 'combo' }, h('b', { class: 'muted' }, i + 1), h('img', { src: '/sprite/' + x.species, style: 'width:46px;height:46px' }),
        h('div', { class: 'col', style: 'flex:1;gap:2px' }, h('b', {}, x.ko), h('span', { class: 'small muted' }, `${x.set.itemKo} · ${x.set.movesKo.join(' / ')}`)),
        h('span', { class: 'pill' }, Math.round(x.score * 100) + '%' + (x.gain != null ? ` (${x.gain >= 0 ? '+' : ''}${Math.round(x.gain * 100)})` : '')),
        h('button', { class: 'btn sm', onclick: () => { const slot = S.team.team.findIndex(s => !s.species); const t = x.set; S.team.team[slot >= 0 ? slot : S.team.team.length] = { species: t.species, ability: t.ability, item: t.item, nature: t.nature, sp: { ...t.sp }, moves: [...t.moves] }; NAME.species[t.species] = t.ko; t.moves.forEach((m, k) => NAME.moves[m] = t.movesKo[k]); NAME.items[t.item] = t.itemKo; saveState(); renderTeam(); toast(x.ko + ' 을(를) 팀에 넣었습니다'); } }, '팀에 넣기'))));
  } catch (e) { res.innerHTML = ''; res.append(h('div', { class: 'empty' }, e.message)); }
});

// 선출 시뮬레이션
$('#bringSimBtn').addEventListener('click', async () => {
  const team = S.team.team.filter(s => s.species);
  const opp = S.oppPreview.filter(Boolean);
  if (team.length < 4 || !opp.length) return toast('내 팀과 상대 6마리를 먼저 넣으세요');
  const el = $('#bringSimRes');
  try {
    const { job } = await api('/api/bring/sim', { team, oppPreview: opp, gameType: S.gt });
    el.innerHTML = '<div class="empty">승률표를 채우는 중… (15~60초)</div>';
    const r = await waitJob(job);
    const lead = S.gt === 'doubles' ? 2 : 1;
    const imgs = (o) => h('div', { class: 'imgs' }, ...o.split('').map((c, j) => h('img', { class: j < lead ? 'lead' : '', src: '/sprite/' + team[+c - 1].species })));
    el.innerHTML = '';
    el.append(h('div', { class: 'grid g2' },
      h('div', { class: 'card', style: 'padding:12px' }, h('div', { class: 'small muted' }, '상대가 메타대로 낼 때 가장 좋은 선출'), h('div', { class: 'row' }, imgs(r.bestMeta.order), h('b', {}, r.bestMeta.names.join(' · ')), h('span', { class: 'pill right' }, Math.round(r.bestMeta.vsMeta * 100) + '%'))),
      h('div', { class: 'card', style: 'padding:12px' }, h('div', { class: 'small muted' }, '상대가 뭘 내든 가장 안전한 선출 (최악의 경우가 가장 나은)'), h('div', { class: 'row' }, imgs(r.bestSafe.order), h('b', {}, r.bestSafe.names.join(' · ')), h('span', { class: 'pill right' }, '최악 ' + Math.round(r.bestSafe.worst * 100) + '%')))),
      h('div', { class: 'small muted', style: 'margin:10px 0' }, `대전 ${r.games}판 · 균형 승률 ${Math.round(r.value * 100)}% · 상대가 낼 법한 선출: ${r.oppLikely.slice(0, 3).map(o => o.names.join('·') + ' ' + Math.round(o.prob * 100) + '%').join(' / ')}`),
      h('table', { class: 't' }, h('tr', {}, ...['선출 (테두리 = 선봉)', '균형 비중', '메타 상대', '최악', '최악일 때 상대'].map(x => h('th', {}, x))),
        ...r.rows.map(x => h('tr', {}, h('td', {}, h('div', { class: 'row' }, imgs(x.order), h('span', { class: 'small' }, x.names.join(' · ')))), h('td', {}, Math.round(x.nash * 100) + '%'),
          h('td', {}, Math.round(x.vsMeta * 100) + '%'), h('td', {}, Math.round(x.worst * 100) + '%'), h('td', { class: 'small muted' }, x.worstVs.join(' · '))))));
  } catch (e) { el.innerHTML = ''; el.append(h('div', { class: 'empty' }, e.message)); }
});

// ======================================================================== 설정
async function renderSettings() {
  const m = await api('/api/meta');
  const box = $('#modBox'); box.innerHTML = '';
  for (const d of m.dataDirs) {
    const f = (m.formats || []).filter(x => x.mod === d.mod).map(x => x.name.replace('[Gen 9 Champions] ', '')).slice(0, 4).join(', ');
    box.append(h('label', { class: 'chk', style: 'padding:6px 0' }, h('input', { type: 'radio', name: 'mod', checked: d.mod === m.mod, onchange: async () => { await api('/api/settings', { mod: d.mod }); toast('규칙을 바꿨습니다'); loadMeta(); } }),
      h('b', {}, d.mod), h('span', { class: 'small muted' }, ` 데이터 ${((d.meta.builtAt || '')).slice(0, 10)} · 쇼다운 ${d.meta.psVersion || ''} ${d.meta.psCommit ? d.meta.psCommit.slice(0, 7) : ''}`)));
  }
  const s = await api('/api/sprites/status'); $('#spriteMsg').textContent = `저장된 그림 ${s.cached}장`;
}
$('#updCheck').addEventListener('click', async () => {
  $('#updMsg').textContent = '확인 중…';
  try {
    const r = await api('/api/update/check', {});
    $('#updMsg').textContent = r.update_available ? `새로 받을 파일 ${r.changed.length}개 (v${r.local} → v${r.remote})` : `최신입니다 (v${r.remote})`;
    $('#updApply').disabled = !r.update_available;
  } catch (e) { $('#updMsg').textContent = e.message; }
});
$('#updApply').addEventListener('click', async () => {
  $('#updMsg').textContent = '받는 중…';
  try { const r = await api('/api/update/apply', {}); $('#updMsg').textContent = `업데이트 완료 (${r.changed.length}개). 프로그램을 다시 실행하면 새 코드가 적용됩니다.`; loadMeta(); }
  catch (e) { $('#updMsg').textContent = e.message; }
});
$('#spriteAll').addEventListener('click', async () => {
  await api('/api/sprites/download', {});
  const poll = async () => { const r = await api('/api/sprites/status'); $('#spriteMsg').textContent = `${r.job.done}/${r.job.total} (실패 ${r.job.failed}) · 저장 ${r.cached}장`; if (r.job.running) setTimeout(poll, 1000); };
  poll();
});
function applyTheme(t) { if (t === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', t); store.set('theme', t); $$('#themeSeg button').forEach(b => b.classList.toggle('on', b.dataset.th === t)); }
$$('#themeSeg button').forEach(b => b.addEventListener('click', () => applyTheme(b.dataset.th)));


// ======================================================================== AI와 대전
const PLAY = { st: null, busy: false, pick: [], sel: {}, mega: {}, teamMode: store.get('playTeam', 'team'), levels: null };
$$('#playTeamSeg button').forEach(b => {
  b.classList.toggle('on', b.dataset.t === PLAY.teamMode);
  b.addEventListener('click', () => { PLAY.teamMode = b.dataset.t; store.set('playTeam', PLAY.teamMode); $$('#playTeamSeg button').forEach(x => x.classList.toggle('on', x === b)); });
});
async function renderPlay() {
  if (!PLAY.levels) {
    try {
      const r = await api('/api/play/levels');
      PLAY.levels = r;
      const sel = $('#playAi'); sel.innerHTML = '';
      for (const [k, v] of r.levels) sel.append(h('option', { value: k, selected: k === store.get('playAi', 'search') }, v));
      sel.onchange = () => store.set('playAi', sel.value);
      if (!r.sim) { $('#playBody').innerHTML = ''; $('#playBody').append(h('div', { class: 'card' }, h('div', { class: 'empty' }, 'AI 와 대전하려면 Node.js 와 쇼다운이 필요합니다. 밤샘학습.bat 을 한 번 실행하면 준비됩니다.'))); return; }
    } catch (e) { toast(e.message); }
  }
  if (!PLAY.st) { try { const r = await api('/api/play/state'); PLAY.st = r.state; } catch (e) {} }
  drawPlay();
}
$('#playStart').addEventListener('click', async () => {
  if (PLAY.busy) return;
  if (PLAY.st && !PLAY.st.ended && !confirm('진행 중인 대전을 그만두고 새로 시작할까요?')) return;
  const team = PLAY.teamMode === 'team' ? (S.team.team || []).filter(s => s.species && (s.moves || []).length) : [];
  if (PLAY.teamMode === 'team' && team.length < (S.gt === 'doubles' ? 4 : 3)) { toast('팀 탭에 기술까지 정한 포켓몬이 부족합니다 — 메타 팀 무작위로 시작해 보세요'); return; }
  await playCall('/api/play/start', { gameType: S.gt, team, ai: $('#playAi').value || 'search' }, '대전을 준비하고 있습니다…');
});
$('#playForfeit').addEventListener('click', async () => { if (PLAY.st && !PLAY.st.ended && confirm('기권할까요?')) await playCall('/api/play/forfeit', {}, ''); });
async function playCall(url, body, msg) {
  PLAY.busy = true; PLAY.busyMsg = msg; drawPlay();
  try { const r = await api(url, body); PLAY.st = r.state; PLAY.pick = []; PLAY.sel = {}; PLAY.mega = {}; }
  catch (e) { toast(e.message); }
  PLAY.busy = false; drawPlay();
}
function playSend(choice) { return playCall('/api/play/choose', { choice }, 'AI 가 생각하고 있습니다…'); }
function pmonCard(m, mine) {
  const st = [m.statusKo, ...Object.entries(m.boosts || {}).map(([k, v]) => `${k} ${v > 0 ? '+' : ''}${v}`)].filter(Boolean).join(' · ');
  const hpTxt = mine && m.maxhp ? `${m.hp}/${m.maxhp}` : `${Math.round(m.hpPct)}%`;
  const extra = mine ? '' : [m.ability && ('특성 ' + m.ability), m.item && ('도구 ' + m.item), (m.moves || []).length && ('쓴 기술 ' + m.moves.join(', '))].filter(Boolean).join(' · ');
  return h('div', { class: 'card pmon ' + (mine ? 'mine' : 'opp') }, sprite(m.species, 'lg'),
    h('div', { class: 'info' }, h('div', { class: 'row' }, h('span', { class: 'nm' }, m.ko + (m.mega ? ' (메가)' : '')), h('span', { class: 'right small muted' }, hpTxt)),
      h('div', { class: 'hpbar ' + hpClass(m.hpPct) }, h('i', { style: `width:${m.hpPct}%` })),
      st ? h('div', { class: 'meta' }, st) : null, extra ? h('div', { class: 'meta' }, extra) : null));
}
function pbench(list, preview, mine) {
  const row = h('div', { class: 'pbench' });
  const seen = new Set(list.map(m => m.species));
  for (const m of list.filter(x => !x.slot || x.fainted)) {
    row.append(h('div', { class: 'b' + (m.fainted ? ' fnt' : '') }, sprite(m.species, 'sm'), h('span', {}, m.ko), h('span', { class: 'muted' }, m.fainted ? '쓰러짐' : Math.round(m.hpPct) + '%')));
  }
  if (!mine) for (const p of preview || []) {
    if ([...seen].some(s => s === p.species || s.startsWith(p.species))) continue;
    row.append(h('div', { class: 'b unseen', title: '아직 안 나옴 (데려왔는지 모름)' }, sprite(p.species, 'sm'), h('span', {}, p.ko), h('span', { class: 'muted' }, '미등장')));
  }
  return row;
}
function drawPlay() {
  const body = $('#playBody'); const st = PLAY.st;
  $('#playStatus').textContent = PLAY.busy ? (PLAY.busyMsg || '…') : !st ? '대기' : st.ended ? '끝' : `${st.turn}턴 · ${st.aiLabel}`;
  $('#playStart').disabled = PLAY.busy; $('#playForfeit').disabled = PLAY.busy || !st || st.ended;
  if (!st) { if (PLAY.busy) { body.innerHTML = ''; body.append(h('div', { class: 'card' }, h('div', { class: 'empty' }, h('span', { class: 'thinking' }, PLAY.busyMsg)))); } return; }
  body.innerHTML = '';
  const actOpp = st.opp.filter(m => m.slot && !m.fainted).sort((a, b) => a.slot.localeCompare(b.slot));
  const actMe = st.mine.filter(m => m.slot && !m.fainted).sort((a, b) => a.slot.localeCompare(b.slot));
  const left = h('div', { class: 'field' },
    h('div', { class: 'card' }, h('div', { class: 'hd' }, '상대 ', h('span', { class: 'sub' }, st.aiLabel)),
      h('div', { class: 'col', style: 'gap:8px;padding:10px' }, ...(actOpp.length ? actOpp.map(m => pmonCard(m, false)) : [h('div', { class: 'small muted' }, '아직 나오지 않음')])),
      pbench(st.opp, st.oppPreview, false)),
    h('div', { class: 'card' }, h('div', { class: 'hd' }, '나'),
      h('div', { class: 'col', style: 'gap:8px;padding:10px' }, ...(actMe.length ? actMe.map(m => pmonCard(m, true)) : [h('div', { class: 'small muted' }, '선출을 고르세요')])),
      pbench(st.mine, null, true)),
    (st.weather || st.terrain) ? h('div', { class: 'small muted' }, [st.weather && ('날씨 ' + st.weather), st.terrain && ('필드 ' + st.terrain)].filter(Boolean).join(' · ')) : null);
  const mid = h('div', { class: 'card' }, h('div', { class: 'hd' }, st.ended ? '결과' : '내 선택', h('span', { class: 'sub' }, PLAY.busy ? '' : (st.aiSeconds || []).length ? `AI 가 최근 ${st.aiSeconds.slice(-1)[0]}초 생각함` : '')));
  if (PLAY.busy) mid.append(h('div', { class: 'empty' }, h('span', { class: 'thinking' }, PLAY.busyMsg)));
  else if (st.ended) {
    const win = st.winner === 'me';
    mid.append(h('div', { class: 'result ' + (win ? 'win' : 'lose') }, win ? '승리했습니다!' : '패배했습니다'),
      h('div', { class: 'empty small' }, st.saved ? `이 대전은 학습 데이터에 저장되었습니다 (${st.saved}). 다음 학습 때 사람의 선택으로 쓰입니다.` : ''),
      h('div', { class: 'row', style: 'justify-content:center;padding-bottom:16px' }, h('button', { class: 'btn primary', onclick: () => $('#playStart').click() }, '한 판 더')));
  } else if (st.options) mid.append(playOptions(st));
  else mid.append(h('div', { class: 'empty' }, '상대를 기다리는 중…'));
  const logEl = h('div', { class: 'plog' }, ...st.log.map(t => h('div', { class: t.startsWith('──') ? 'turn' : t.startsWith('★') ? 'end' : '' }, t)));
  const right = h('div', { class: 'card logcol' }, h('div', { class: 'hd' }, '대전 기록'), logEl);
  body.append(h('div', { class: 'play' }, left, mid, right));
  logEl.scrollTop = logEl.scrollHeight;
}
function playOptions(st) {
  const o = st.options;
  const box = h('div', { class: 'popts' });
  if (o.kind === 'preview') {
    const need = o.pick, leads = need > 3 ? 2 : 1;
    box.append(h('div', { class: 'small muted' }, `${need}마리를 고르세요. 처음 고른 ${leads}마리가 선봉입니다.`));
    const grid = h('div', { class: 'pick6' });
    for (const m of o.mons) {
      const k = PLAY.pick.indexOf(m.n);
      grid.append(h('div', { class: 'card p pk' + (k >= 0 ? ' on' : ''), onclick: () => {
        if (k >= 0) PLAY.pick.splice(k, 1); else if (PLAY.pick.length < need) PLAY.pick.push(m.n);
        drawPlay();
      } }, k >= 0 ? h('span', { class: 'no' }, k < leads ? '선봉' : String(k + 1)) : null, sprite(m.species), h('b', { class: 'small' }, m.ko),
        h('span', { class: 'small muted' }, m.item || ''), h('span', { class: 'small muted', style: 'text-align:center' }, (m.moves || []).join(' · '))));
    }
    box.append(grid, h('div', { class: 'row' }, h('button', { class: 'btn primary', disabled: PLAY.pick.length !== need, onclick: () => playSend('team ' + PLAY.pick.join('')) }, '이 선출로 시작'),
      h('button', { class: 'btn ghost', onclick: () => { PLAY.pick = []; drawPlay(); } }, '다시 고르기')));
    return box;
  }
  const multi = o.slots.length > 1;
  for (const slot of o.slots) {
    const si = slot.slot;
    const sel = PLAY.sel[si];
    const sec = h('div', { class: 'col', style: 'gap:8px' }, h('div', { class: 'slot-hd' }, (multi ? (si === 0 ? '왼쪽 · ' : '오른쪽 · ') : '') + (o.kind === 'force' ? '교체할 포켓몬' : (slot.who + '의 행동'))));
    const moves = slot.entries.filter(e => e.kind === 'move');
    const canMega = moves.some(e => e.mega);
    if (moves.length) {
      // 같은 기술의 대상별 항목은 하나로 묶는다
      const groups = [];
      for (const e of moves) { const base = e.part.split(' ').slice(0, 2).join(' '); let g = groups.find(x => x.base === base); if (!g) groups.push(g = { base, e, tg: [] }); if (e.target) g.tg.push(e); }
      const grid = h('div', { class: 'mvgrid' });
      for (const g of groups) {
        const on = sel && sel.startsWith(g.base) && !sel.startsWith('switch');
        grid.append(h('button', { class: 'mvbtn' + (on ? ' on' : ''), onclick: () => {
          PLAY.sel[si] = g.tg.length ? g.tg[0].part : g.e.part;
          if (!multi && !g.tg.length) return playSend(PLAY.sel[si] + (PLAY.mega[si] ? ' mega' : ''));
          drawPlay();
        } }, h('span', { class: 't1' }, g.e.type ? typeBadge(g.e.type) : null, g.e.label),
          h('span', { class: 't2' }, [g.e.category === 'Status' ? '변화' : g.e.category === 'Physical' ? '물리' : g.e.category === 'Special' ? '특수' : '', g.e.power ? '위력 ' + g.e.power : '', g.e.pp != null ? `PP ${g.e.pp}/${g.e.maxpp}` : ''].filter(Boolean).join(' · '))));
      }
      sec.append(grid);
      const cur = groups.find(g => sel && sel.startsWith(g.base) && g.tg.length);
      if (cur) sec.append(h('div', { class: 'tgts' }, h('span', { class: 'small muted' }, '대상'), ...cur.tg.map(e => h('button', { class: 'btn sm' + (sel === e.part ? ' primary' : ''), onclick: () => { PLAY.sel[si] = e.part; if (!multi) return playSend(e.part + (PLAY.mega[si] ? ' mega' : '')); drawPlay(); } }, e.target))));
      if (canMega) sec.append(h('label', { class: 'chk small' }, h('input', { type: 'checkbox', checked: !!PLAY.mega[si], onchange: (e) => { PLAY.mega[si] = e.target.checked; if (e.target.checked) for (const k of Object.keys(PLAY.mega)) if (+k !== si) PLAY.mega[k] = false; drawPlay(); } }), '메가진화하면서'));
    }
    const sw = slot.entries.filter(e => e.kind === 'switch');
    if (sw.length) sec.append(h('div', { class: 'small muted' }, o.kind === 'force' ? '' : '교체'), h('div', { class: 'swrow' }, ...sw.map(e => {
      const hp = e.condition.split(' ')[0];
      return h('button', { class: 'swbtn' + (sel === e.part ? ' on' : ''), onclick: () => { PLAY.sel[si] = e.part; if (!multi) return playSend(e.part); drawPlay(); } }, sprite(e.species, 'sm'), e.label, h('span', { class: 'small muted' }, hp));
    })));
    const pass = slot.entries.find(e => e.kind === 'pass');
    if (pass && slot.entries.length === 1) { PLAY.sel[si] = 'pass'; sec.append(h('div', { class: 'small muted' }, '이 자리는 넘깁니다')); }
    else if (pass) sec.append(h('button', { class: 'btn sm ghost', onclick: () => { PLAY.sel[si] = 'pass'; drawPlay(); } }, '넘기기'));
    box.append(sec);
  }
  if (multi) {
    const parts = o.slots.map(sl => { const p = PLAY.sel[sl.slot]; return p && p !== 'pass' && PLAY.mega[sl.slot] && p.startsWith('move') ? p + ' mega' : p; });
    const choice = parts.every(Boolean) ? parts.join(', ') : null;
    const legal = choice && o.joint.includes(choice);
    box.append(h('div', { class: 'row' }, h('button', { class: 'btn primary', disabled: !legal, onclick: () => playSend(choice) }, '이대로 행동'),
      choice && !legal ? h('span', { class: 'small', style: 'color:var(--bad)' }, '같이 고를 수 없는 조합입니다 (같은 포켓몬으로 두 번 교체 등)') : null));
  }
  return box;
}

// ======================================================================== 시작
async function loadMeta() {
  S.meta = await api('/api/meta');
  const d = S.meta.data;
  $('#dataVer').textContent = `${S.meta.mod} · 데이터 ${(d.builtAt || '').slice(0, 10)} · v${S.meta.version}`;
}
(async function init() {
  applyTheme(store.get('theme', 'auto'));
  $$('#gtSeg button').forEach(b => b.classList.toggle('on', b.dataset.gt === S.gt));
  await loadMeta();
  await renderTeam(); renderTeamList(); renderBring();
  showPage(store.get('page', 'battle'));
})();

#!/usr/bin/env node
/*
 * 포켓몬 쇼다운의 챔피언스 모드(data/mods/champions) 데이터를 JSON 으로 뽑는다.
 *
 *   node tools/export_champions.js <쇼다운 폴더> <출력 폴더>
 *
 * <쇼다운 폴더> 는 빌드된 pokemon-showdown (dist/ 가 있는 폴더).
 * GitHub Actions(update-data.yml)가 쇼다운 최신 master 로 매일 돌린다.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const psDir = path.resolve(process.argv[2] || 'pokemon-showdown');
const outDir = path.resolve(process.argv[3] || 'build/raw');
const MOD = process.env.PS_MOD || 'champions';

const { Dex } = require(path.join(psDir, 'dist', 'sim'));
const dex = Dex.mod(MOD);
fs.mkdirSync(outDir, { recursive: true });

const legal = x => x.exists && !x.isNonstandard;
const plain = v => JSON.parse(JSON.stringify(v, (k, val) => (typeof val === 'function' ? undefined : val)));
const fnKeys = obj => Object.keys(obj).filter(k => typeof obj[k] === 'function').sort();

// ---------------------------------------------------------------- 포켓몬
const species = {};
for (const s of dex.species.all()) {
	if (!legal(s) || s.tier === 'Illegal') continue;
	species[s.id] = {
		id: s.id, name: s.name, num: s.num, types: s.types, baseStats: s.baseStats,
		abilities: s.abilities, weightkg: s.weightkg, baseSpecies: s.baseSpecies,
		forme: s.forme || '', isMega: !!s.isMega, requiredItem: s.requiredItem || '',
		changesFrom: s.changesFrom || '', spriteid: s.spriteid, tier: s.tier,
		nfe: !!s.nfe, gender: s.gender || '', otherFormes: s.otherFormes || [],
	};
}

// ---------------------------------------------------------------- 배울 수 있는 기술
const learnsets = {};
for (const id of Object.keys(species)) {
	const pool = new Set();
	let cur = dex.species.get(id);
	const seen = new Set();
	// 메가·폼은 원래 폼의 기술을 쓰고, 진화 전 기술도 물려받는다
	while (cur && cur.exists && !seen.has(cur.id)) {
		seen.add(cur.id);
		const ls = dex.data.Learnsets[cur.id];
		if (ls && ls.learnset) for (const m of Object.keys(ls.learnset)) pool.add(m);
		const next = (!ls || !ls.learnset) ? (cur.changesFrom || cur.baseSpecies) : cur.prevo;
		cur = next ? dex.species.get(next) : null;
	}
	learnsets[id] = [...pool].filter(m => legal(dex.moves.get(m))).sort();
}

// ---------------------------------------------------------------- 기술
const MOVE_KEYS = [
	'num', 'name', 'type', 'category', 'basePower', 'accuracy', 'pp', 'priority', 'target', 'flags',
	'secondary', 'secondaries', 'drain', 'recoil', 'heal', 'multihit', 'multiaccuracy', 'critRatio',
	'willCrit', 'ohko', 'damage', 'ignoreDefensive', 'ignoreEvasion', 'ignoreImmunity', 'ignoreAbility',
	'overrideOffensiveStat', 'overrideOffensivePokemon', 'overrideDefensiveStat', 'boosts', 'self',
	'selfBoost', 'status', 'volatileStatus', 'sideCondition', 'slotCondition', 'weather', 'terrain',
	'pseudoWeather', 'selfSwitch', 'forceSwitch', 'breaksProtect', 'hasCrashDamage', 'mindBlownRecoil',
	'stallingMove', 'sleepUsable', 'thawsTarget', 'selfdestruct', 'smartTarget', 'tracksTarget',
	'hasSheerForce', 'struggleRecoil', 'noPPBoosts', 'shortDesc', 'desc',
];
const moves = {};
for (const m of dex.moves.all()) {
	if (!legal(m) || m.isZ || m.isMax) continue;
	const o = { id: m.id };
	for (const k of MOVE_KEYS) if (m[k] !== undefined) o[k] = plain(m[k]);
	o.maxpp = m.noPPBoosts ? m.pp : Math.floor((m.pp / 5 + 1) * 4);  // champions calculatePP
	o.callbacks = fnKeys(m);
	if (m.secondaries) {
		o.secondaryCallbacks = m.secondaries.some(s => fnKeys(s).length || (s.self && fnKeys(s.self).length));
	}
	moves[m.id] = o;
}

// ---------------------------------------------------------------- 도구
const items = {};
for (const it of dex.items.all()) {
	if (!legal(it)) continue;
	items[it.id] = {
		id: it.id, name: it.name, num: it.num, megaStone: it.megaStone || null,
		isBerry: !!it.isBerry, naturalGift: it.naturalGift || null, fling: it.fling || null,
		shortDesc: it.shortDesc || '', desc: it.desc || '', callbacks: fnKeys(it),
	};
}

// ---------------------------------------------------------------- 특성
const abilities = {};
for (const a of dex.abilities.all()) {
	if (!legal(a) || a.id === 'noability') continue;
	abilities[a.id] = {
		id: a.id, name: a.name, num: a.num, flags: plain(a.flags || {}), rating: a.rating,
		shortDesc: a.shortDesc || '', desc: a.desc || '', callbacks: fnKeys(a),
	};
}

// ---------------------------------------------------------------- 타입·성격
const typechart = {};
for (const t of dex.types.all()) {
	if (t.isNonstandard) continue;
	typechart[t.name] = { damageTaken: t.damageTaken };
}
const natures = {};
for (const n of dex.natures.all()) natures[n.id] = { name: n.name, plus: n.plus || null, minus: n.minus || null };

// ---------------------------------------------------------------- 랜덤배틀 세트 (상대 세트 추정의 사전확률)
let sets = {};
for (const p of [
	path.join(psDir, 'dist', 'data', 'random-battles', MOD, 'sets.json'),
	path.join(psDir, 'data', 'random-battles', MOD, 'sets.json'),
]) {
	if (fs.existsSync(p)) { sets = JSON.parse(fs.readFileSync(p, 'utf8')); break; }
}

// ---------------------------------------------------------------- 포맷
const formats = Dex.formats.all()
	.filter(f => f.mod && f.mod.startsWith('champions') && f.effectType === 'Format')
	.map(f => ({ id: f.id, name: f.name, mod: f.mod, gameType: f.gameType || 'singles' }));

let psVersion = '';
try { psVersion = require(path.join(psDir, 'package.json')).version; } catch (e) {}

const out = { species, learnsets, moves, items, abilities, typechart, natures, sets, formats };
for (const [k, v] of Object.entries(out)) {
	fs.writeFileSync(path.join(outDir, k + '.json'), JSON.stringify(v));
}
fs.writeFileSync(path.join(outDir, 'meta.json'), JSON.stringify({
	mod: MOD, psVersion, psCommit: process.env.PS_COMMIT || '', exportedAt: new Date().toISOString(),
	counts: Object.fromEntries(Object.entries(out).map(([k, v]) => [k, Object.keys(v).length])),
}, null, 1));
console.log('exported', Object.fromEntries(Object.entries(out).map(([k, v]) => [k, Object.keys(v).length])));

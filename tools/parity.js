#!/usr/bin/env node
/*
 * 데미지 대조 사례 생성기 — 쇼다운 시뮬레이터(챔피언스 모드)가 계산한 정답을 만든다.
 *
 *   node tools/parity.js <쇼다운 폴더> <출력.json> [사례 수] [시드] [singles|doubles|mixed]
 *
 * 무작위로 포켓몬·특성·도구·성격·SP·기술·랭크·상태이상·HP·날씨·필드·벽·도우미 등을 고르고,
 * 쇼다운 getDamage 를 난수 16단계 × (급소 아님/급소) 로 불러 결과를 기록한다.
 * tests/selftest.py 가 같은 사례를 파이썬 엔진으로 계산해 한 칸씩 비교한다.
 */
'use strict';
const path = require('path');
const fs = require('fs');

const psDir = path.resolve(process.argv[2] || 'pokemon-showdown');
const outFile = process.argv[3] || 'tests/parity.json';
const N = parseInt(process.argv[4] || '2000', 10);
let seed = parseInt(process.argv[5] || '1', 10);
const MODE = process.argv[6] || 'mixed';
const MOD = process.env.PS_MOD || 'champions';

const { Dex, Battle, Teams } = require(path.join(psDir, 'dist', 'sim'));
const dex = Dex.mod(MOD);
const tr = (n) => n >>> 0;

// ---------------------------------------------------------------- 결정적 난수
function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x80000000; }
const pick = (a) => a[Math.floor(rnd() * a.length)];
const chance = (p) => rnd() < p;
const randInt = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));

// ---------------------------------------------------------------- 후보
const species = dex.species.all().filter(s => s.exists && !s.isNonstandard && s.tier !== 'Illegal');
const AVOID_AB = new Set(['imposter', 'trace', 'illusion', 'zenmode', 'stancechange', 'forecast', 'schooling',
	'powerconstruct', 'shieldsdown', 'commander', 'neutralizinggas', 'mimicry', 'zerotohero', 'flowergift',
	'hungerswitch', 'gulpmissile', 'iceface', 'disguise', 'battlebond', 'multitype', 'rkssystem', 'asone',
	'protosynthesis', 'quarkdrive', 'orichalcumpulse', 'hadronengine', 'teraformzero', 'terashift']);
const items = dex.items.all().filter(i => i.exists && !i.isNonstandard).map(i => i.id);
const COMMON_ITEMS = ['lifeorb', 'expertbelt', 'muscleband', 'wiseglasses', 'choicescarf', 'focussash',
	'leftovers', 'sitrusberry', 'charcoal', 'mysticwater', 'magnet', 'miracleseed', 'occaberry',
	'chilanberry', 'shucaberry', 'yacheberry', 'lightball', 'scopelens', 'ironball', ''];
const natures = dex.natures.all().map(n => n.id);
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const WEATHERS = ['', '', '', 'sunnyday', 'raindance', 'sandstorm', 'snowscape'];
const TERRAINS = ['', '', '', 'electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain'];
const STATUSES = ['', '', '', '', 'brn', 'par', 'psn', 'tox', 'slp'];

function learnable(sp) {
	const pool = new Set();
	let cur = sp; const seen = new Set();
	while (cur && cur.exists && !seen.has(cur.id)) {
		seen.add(cur.id);
		const ls = dex.data.Learnsets[cur.id];
		if (ls && ls.learnset) for (const m of Object.keys(ls.learnset)) pool.add(m);
		const nx = (!ls || !ls.learnset) ? (cur.changesFrom || cur.baseSpecies) : cur.prevo;
		cur = nx ? dex.species.get(nx) : null;
	}
	return [...pool].map(m => dex.moves.get(m)).filter(m => m.exists && !m.isNonstandard && m.category !== 'Status');
}

function randomSP() {
	const sp = {}; let left = 66;
	for (const s of STATS) sp[s] = 0;
	const order = STATS.slice().sort(() => rnd() - 0.5);
	for (const s of order) { const v = Math.min(32, left, randInt(0, 32)); sp[s] = chance(0.3) ? 0 : v; left -= sp[s]; }
	return sp;
}

function randomMon(forceMoves) {
	const sp = pick(species);
	const abs = Object.values(sp.abilities).map(a => dex.abilities.get(a).id).filter(a => !AVOID_AB.has(a));
	if (!abs.length) return null;
	let item = chance(0.7) ? pick(COMMON_ITEMS) : pick(items);
	if (sp.requiredItem) item = dex.items.get(sp.requiredItem).id;
	const moves = learnable(sp);
	if (forceMoves && !moves.length) return null;
	return {
		species: sp.id, name: sp.name, ability: pick(abs), item, nature: pick(natures), sp: randomSP(),
		gender: sp.gender ? sp.gender : pick(['M', 'F']), moves,
	};
}

function boosts() {
	const b = {};
	if (chance(0.4)) for (const s of ['atk', 'def', 'spa', 'spd', 'spe']) if (chance(0.4)) b[s] = randInt(-3, 4);
	return b;
}

function setFromMon(m) {
	return {
		name: m.name, species: m.name, item: m.item, ability: m.ability, nature: m.nature,
		evs: m.sp, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, gender: m.gender === 'N' ? '' : m.gender,
		moves: ['splash'], level: 50,
	};
}

function makeCase(i) {
	const doubles = MODE === 'doubles' || (MODE === 'mixed' && chance(0.4));
	const A = randomMon(true), D = randomMon(false);
	if (!A || !D) return null;
	const AA = doubles ? randomMon(false) : null, DA = doubles ? randomMon(false) : null;
	if (doubles && (!AA || !DA)) return null;
	const move = pick(A.moves);
	const cs = {
		id: i, gameType: doubles ? 'doubles' : 'singles', move: move.id,
		field: {
			weather: pick(WEATHERS), terrain: pick(TERRAINS), gravity: chance(0.05), trickRoom: chance(0.15),
		},
		sides: [
			{ reflect: false, lightscreen: false, auroraveil: false, tailwind: chance(0.15), fainted: chance(0.2) ? randInt(1, 2) : 0 },
			{ reflect: chance(0.15), lightscreen: chance(0.15), auroraveil: chance(0.08), tailwind: chance(0.15), fainted: 0 },
		],
		helpingHand: doubles && chance(0.25),
		mons: {},
	};
	const states = { atk: A, def: D, atkAlly: AA, defAlly: DA };
	for (const [k, m] of Object.entries(states)) {
		if (!m) continue;
		m.status = pick(STATUSES);
		m.boosts = boosts();
		m.hpFrac = chance(0.5) ? 1 : Math.max(0.01, rnd());
		m.timesAttacked = chance(0.1) ? randInt(1, 3) : 0;
		m.flashfire = m.ability === 'flashfire' && chance(0.5);
	}
	return { cs, states };
}

function runCase(i) {
	const made = makeCase(i);
	if (!made) return null;
	const { cs, states } = made;
	const doubles = cs.gameType === 'doubles';
	const formatid = doubles ? 'gen9championsdoublescustomgame' : 'gen9championscustomgame';
	const p1 = [setFromMon(states.atk)]; if (doubles) p1.push(setFromMon(states.atkAlly));
	const p2 = [setFromMon(states.def)]; if (doubles) p2.push(setFromMon(states.defAlly));
	let battle;
	try {
		battle = new Battle({ formatid, seed: [i, 2, 3, 4] });
		battle.setPlayer('p1', { name: 'A', team: Teams.pack(p1) });
		battle.setPlayer('p2', { name: 'B', team: Teams.pack(p2) });
		if (battle.requestState === 'teampreview') {
			battle.choose('p1', doubles ? 'team 12' : 'team 1');
			battle.choose('p2', doubles ? 'team 12' : 'team 1');
		}
	} catch (e) {
		return null;
	}
	const f = battle.field;
	const [atk, atkAlly] = battle.sides[0].active;
	const [def, defAlly] = battle.sides[1].active;
	if (!atk || !def) return null;
	const mons = { atk, def, atkAlly: doubles ? atkAlly : null, defAlly: doubles ? defAlly : null };

	// ---- 상태 덮어쓰기 (등장 특성으로 바뀐 것 되돌림)
	f.clearWeather(); f.clearTerrain();
	if (cs.field.weather) { f.setWeather(cs.field.weather, atk); f.weatherState.duration = 0; }
	if (cs.field.terrain) { f.setTerrain(cs.field.terrain, atk); f.terrainState.duration = 0; }
	for (const pw of ['gravity', 'trickroom']) f.removePseudoWeather(pw);
	if (cs.field.gravity) f.addPseudoWeather('gravity', atk);
	if (cs.field.trickRoom) f.addPseudoWeather('trickroom', atk);
	for (let s = 0; s < 2; s++) {
		const side = battle.sides[s];
		for (const sc of ['reflect', 'lightscreen', 'auroraveil', 'tailwind']) {
			side.removeSideCondition(sc);
			if (cs.sides[s][sc]) side.addSideCondition(sc, side.active[0], null);
		}
		side.totalFainted = cs.sides[s].fainted;
	}
	for (const [k, p] of Object.entries(mons)) {
		if (!p) continue;
		const st = states[k];
		p.boosts = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0, ...st.boosts };
		p.hp = Math.max(1, Math.floor(p.maxhp * st.hpFrac));
		p.status = st.status; p.statusState = battle.initEffectState({ id: st.status, target: p });
		p.timesAttacked = st.timesAttacked;
		for (const v of Object.keys(p.volatiles)) p.removeVolatile(v);
		if (st.flashfire) p.addVolatile('flashfire');
		if (p.ability === 'supremeoverlord') p.abilityState.fallen = Math.min(5, p.side.totalFainted);
		st.out = {
			species: p.species.id, ability: p.ability, item: p.item, nature: st.nature, sp: st.sp,
			gender: p.gender || '', hp: p.hp, status: p.status, boosts: st.boosts,
			volatiles: st.flashfire ? ['flashfire'] : [], timesAttacked: p.timesAttacked,
			stats: { hp: p.maxhp, ...p.storedStats },
		};
	}
	if (cs.helpingHand) atk.addVolatile('helpinghand', atkAlly);
	battle.updateSpeed();

	// ---- 기술 준비 (useMoveInner 의 ModifyType/ModifyMove)
	let move = battle.dex.getActiveMove(cs.move);
	battle.activePokemon = atk; battle.activeTarget = def; battle.activeMove = move;
	battle.setActiveMove(move, atk, def);
	battle.singleEvent('ModifyType', move, null, atk, def, move, move);
	battle.singleEvent('ModifyMove', move, null, atk, def, move, move);
	move = battle.runEvent('ModifyType', atk, def, move, move);
	move = battle.runEvent('ModifyMove', atk, def, move, move);
	if (doubles && ['allAdjacentFoes', 'allAdjacent'].includes(move.target)) {
		let n = 1 + (defAlly && defAlly.hp ? 1 : 0);
		if (move.target === 'allAdjacent' && atkAlly && atkAlly.hp) n++;
		move.spreadHit = n > 1;
	}
	battle.randomChance = () => false;
	const origItem = def.item;
	const nHits = move.id === 'tripleaxel' ? 3 : 1;
	const result = { normal: [], crit: [] };
	for (const crit of [false, true]) {
		for (let h = 1; h <= nHits; h++) {
			const row = [];
			for (let r = 0; r < 16; r++) {
				def.item = origItem; def.itemState = battle.initEffectState({ id: origItem, target: def });
				move.hit = h; move.willCrit = crit;
				battle.randomizer = (bd) => tr(tr(bd * (100 - r)) / 100);
				let d;
				try { d = battle.actions.getDamage(atk, def, move, true); } catch (e) { return null; }
				row.push(d === false ? false : d === undefined || d === null ? 0 : d);
			}
			(crit ? result.crit : result.normal).push(row);
		}
	}
	return {
		id: cs.id, gameType: cs.gameType, move: cs.move, moveType: move.type, category: move.category,
		field: { ...cs.field }, sides: cs.sides, helpingHand: cs.helpingHand, spreadHit: !!move.spreadHit,
		// 이 시점의 쇼다운 상태: 행동 대기열이 비어 있음 → 애널라이즈 발동, 보복은 상대가 먼저 움직인 것으로 계산
		ctx: { movedLast: true, targetMovedFirst: !def.newlySwitched },
		atk: states.atk.out, def: states.def.out,
		atkAlly: doubles ? states.atkAlly.out : null, defAlly: doubles ? states.defAlly.out : null,
		expect: result,
	};
}

const cases = [];
let tries = 0;
while (cases.length < N && tries < N * 5) {
	tries++;
	const c = runCase(tries);
	if (c) cases.push(c);
}
let psVersion = '';
try { psVersion = require(path.join(psDir, 'package.json')).version; } catch (e) {}
fs.mkdirSync(path.dirname(path.resolve(outFile)), { recursive: true });
fs.writeFileSync(outFile, JSON.stringify({ mod: MOD, psVersion, mode: MODE, n: cases.length, cases }));
console.log(`parity cases: ${cases.length} (tries ${tries}) → ${outFile}`);

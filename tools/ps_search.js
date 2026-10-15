/*
 * 수읽기(탐색) — 쇼다운 시뮬레이터로 몇 턴 앞을 직접 굴려 행동별 승률을 잰다.
 *
 * 불완전 정보: 파이썬(engine/search.py)이 상대의 보이지 않는 정보(기술·도구·SP·아직 안 나온 포켓몬)를
 * 추정 확률대로 K 번 뽑아 '가능한 세계'를 만들어 보낸다. 여기서는 세계마다, 행동마다 R 번씩
 * D 턴을 굴린 뒤 판세를 평가해 평균한다 (정보집합 몬테카를로 탐색).
 */
'use strict';

module.exports = function makeSearch(Battle) {
	const range = (n) => Array.from({ length: n }, (_, i) => i + 1).join('');

	// ---------------------------------------------------------------- 장면 만들기
	function build(sc, world, seed) {
		const fmt = sc.gameType === 'doubles' ? 'gen9championsdoublescustomgame' : 'gen9championscustomgame';
		const b = new Battle({ formatid: fmt, seed });
		const sides = { p1: sc.p1, p2: sc.p2 };
		sides[sc.oppSide] = world;
		b.setPlayer('p1', { name: 'p1', team: sides.p1.packed });
		b.setPlayer('p2', { name: 'p2', team: sides.p2.packed });
		b.choose('p1', 'team ' + range(sides.p1.state.length));
		b.choose('p2', 'team ' + range(sides.p2.state.length));
		const f = b.field;
		const src = b.sides[0].active[0];
		f.clearWeather(); f.clearTerrain();
		if (sc.field.weather) { f.setWeather(sc.field.weather, src); f.weatherState.duration = Math.max(1, 5 - (sc.field.weatherTurns || 0)); }
		if (sc.field.terrain) { f.setTerrain(sc.field.terrain, src); f.terrainState.duration = Math.max(1, 5 - (sc.field.terrainTurns || 0)); }
		for (const pw of ['gravity', 'trickroom']) f.removePseudoWeather(pw);
		if (sc.field.trickRoom) f.addPseudoWeather('trickroom', src);
		if (sc.field.gravity) f.addPseudoWeather('gravity', src);
		for (let s = 0; s < 2; s++) {
			const side = b.sides[s];
			const st = sides[side.id];
			for (const sc_ of ['reflect', 'lightscreen', 'auroraveil', 'tailwind', 'stealthrock', 'spikes', 'toxicspikes', 'stickyweb']) {
				side.removeSideCondition(sc_);
			}
			for (const [k, v] of Object.entries((sc.sides || {})[side.id] || {})) {
				if (!v) continue;
				const n = typeof v === 'number' ? v : 1;
				for (let i = 0; i < n; i++) side.addSideCondition(k, side.active[0], null);
			}
			let left = 0;
			side.pokemon.forEach((p, i) => {
				const ps = st.state[i];
				if (!ps) return;
				p.boosts = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0, ...(ps.boosts || {}) };
				for (const v of Object.keys(p.volatiles)) p.removeVolatile(v);
				if (ps.fainted) {
					p.hp = 0; p.fainted = true; p.status = 'fnt';
				} else {
					p.hp = Math.max(1, ps.hp != null ? Math.min(p.maxhp, ps.hp) : Math.round(p.maxhp * (ps.hpPct ?? 100) / 100));
					p.status = ps.status || ''; p.statusState = b.initEffectState({ id: p.status, target: p });
					if (p.status === 'slp') p.statusState.time = 2;
					left++;
				}
				if (ps.noItem) { p.item = ''; }
				if (!ps.fainted && p.status === 'tox') p.statusState.stage = Math.max(0, Math.min(15, ps.toxStage || 1));
			});
			side.pokemonLeft = left;
			if ((sc.megaUsed || {})[side.id]) for (const p of side.pokemon) p.canMegaEvo = null;
		}
		b.turn = sc.turn || b.turn;
		b.updateSpeed();
		b.log = []; b.sentLogPos = 0;
		return b;
	}

	// ---------------------------------------------------------------- 빠른 굴리기 정책
	function estFrac(b, atk, def, move) {
		if (!move || move.category === 'Status' || !def || def.fainted) return 0;
		if (!def.runImmunity(move.type)) return 0;
		const phys = move.category === 'Physical';
		let bp = move.basePower || 60;
		// 특성·도구 배율 (테크니션 + 불릿펀치 등) — 빠른 추정이라 대표적인 것만
		const fl = move.flags || {};
		if (atk.hasAbility('technician') && bp <= 60) bp *= 1.5;
		if (atk.hasAbility('toughclaws') && fl.contact) bp *= 1.3;
		if (atk.hasAbility('ironfist') && fl.punch) bp *= 1.2;
		if (atk.hasAbility('sharpness') && fl.slicing) bp *= 1.5;
		if (atk.hasAbility('strongjaw') && fl.bite) bp *= 1.5;
		if (atk.hasAbility('megalauncher') && fl.pulse) bp *= 1.5;
		if (atk.hasAbility('sheerforce') && (move.secondary || move.secondaries)) bp *= 1.3;
		if (move.multihit) bp *= Array.isArray(move.multihit) ? (atk.hasAbility('skilllink') ? move.multihit[1] : 3) : move.multihit;
		let A = atk.getStat(phys ? 'atk' : 'spa', false, true);
		if (phys && (atk.hasAbility('hugepower') || atk.hasAbility('purepower'))) A *= 2;
		if (atk.hasItem(phys ? 'choiceband' : 'choicespecs')) A *= 1.5;
		const D = Math.max(1, def.getStat(phys ? 'def' : 'spd', false, true));
		let d = (Math.floor(Math.floor(22 * bp * A / D) / 50) + 2) * 0.925;
		if (atk.hasType(move.type)) d *= atk.hasAbility('adaptability') ? 2 : 1.5;
		const eff = b.dex.getEffectiveness(move.type, def);
		d *= Math.pow(2, eff);
		if (atk.hasItem('lifeorb')) d *= 1.3;
		if (eff > 0 && atk.hasItem('expertbelt')) d *= 1.2;
		if (phys && atk.status === 'brn') d *= 0.5;
		const acc = move.accuracy === true ? 1 : (move.accuracy || 100) / 100;
		return (d / def.maxhp) * acc;
	}

	const SETUP = new Set(['swordsdance', 'nastyplot', 'dragondance', 'calmmind', 'bulkup', 'quiverdance', 'shellsmash', 'coil', 'irondefense', 'agility', 'victorydance', 'tidyup', 'bellydrum']);
	const PROTECT = new Set(['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'silktrap', 'burningbulwark']);
	const RECOVER = new Set(['recover', 'roost', 'slackoff', 'softboiled', 'moonlight', 'morningsun', 'synthesis', 'milkdrink', 'shoreup', 'strengthsap']);
	const ANTI_SETUP = new Set(['haze', 'clearsmog', 'roar', 'whirlwind', 'dragontail', 'circlethrow', 'encore', 'spectralthief', 'topsyturvy']);
	const HAZARD_MOVES = new Set(['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']);

	function bestVs(b, atk, def) {
		let best = 0;
		for (const ms of atk.moveSlots) best = Math.max(best, estFrac(b, atk, def, b.dex.moves.get(ms.id)));
		return best;
	}

	function rolloutChoice(b, side, rng, temp) {
		const req = side.activeRequest;
		if (!req || req.wait) return null;
		const foes = side.foe.active.filter(p => p && !p.fainted);
		const bench = side.pokemon.map((p, i) => ({ p, i: i + 1 })).filter(x => !x.p.isActive && !x.p.fainted);
		const pickSoft = (cands) => {
			if (!cands.length) return null;
			const mx = Math.max(...cands.map(c => c.s));
			const w = cands.map(c => Math.exp((c.s - mx) / temp));
			let r = rng() * w.reduce((a, b) => a + b, 0);
			for (let i = 0; i < cands.length; i++) { r -= w[i]; if (r <= 0) return cands[i]; }
			return cands[cands.length - 1];
		};
		const matchup = (p) => {
			let v = 0;
			for (const f of foes) v += bestVs(b, p, f) - 1.2 * bestVs(b, f, p);
			return foes.length ? v / foes.length : 0;
		};
		if (req.forceSwitch) {
			const used = new Set();
			return req.forceSwitch.map((need) => {
				if (!need) return 'pass';
				const c = pickSoft(bench.filter(x => !used.has(x.i)).map(x => ({ x, s: matchup(x.p) })));
				if (!c) return 'pass';
				used.add(c.x.i);
				return 'switch ' + c.x.i;
			}).join(', ');
		}
		const doubles = side.active.length > 1;
		const parts = [];
		const usedSw = new Set();
		let megaDone = false;
		req.active.forEach((a, si) => {
			const me = side.active[si];
			if (!me || me.fainted) { parts.push('pass'); return; }
			const cands = [];
			const hpf = me.hp / me.maxhp;
			const threat = foes.reduce((m, f) => Math.max(m, bestVs(b, f, me)), 0);
			a.moves.forEach((mv, j) => {
				if (mv.disabled || mv.pp === 0) return;
				const move = b.dex.moves.get(mv.id);
				if (move.category === 'Status') {
					let s = 0.05;
					const myBoost = ['atk', 'spa', 'spe', 'def', 'spd'].reduce((t, k) => t + Math.max(0, me.boosts[k]), 0);
					const foeBoost = Math.max(0, ...foes.map(f => ['atk', 'spa', 'spe'].reduce((t, k) => t + Math.max(0, f.boosts[k]), 0)));
					const foeStatus = foes.some(f => f.status && f.status !== 'fnt');
					const foeSub = foes.some(f => f.volatiles.substitute);
					if (PROTECT.has(move.id)) s = doubles ? 0.25 : (foeStatus || me.volatiles.wish ? 0.3 : 0.02);
					else if (SETUP.has(move.id)) s = hpf > 0.7 && threat < 0.4 ? 0.45 : 0.02;
					else if (RECOVER.has(move.id) || move.id === 'wish') s = hpf < 0.5 ? 0.5 : 0.0;
					else if (move.id === 'batonpass' || move.id === 'shedtail') s = (myBoost >= 2 || move.id === 'shedtail') && bench.length && (threat >= 0.45 || hpf < 0.5) ? 0.8 : 0.0;
					else if (move.id === 'substitute') s = !me.volatiles.substitute && hpf > 0.5 && threat < 0.25 ? 0.4 : 0.0;
					else if (ANTI_SETUP.has(move.id)) s = foeBoost >= 2 ? 0.9 : 0.02;
					else if (move.status || move.id === 'toxic') s = !foeSub && foes.some(f => !f.status && f.runStatusImmunity(move.status || 'tox')) ? (move.status === 'slp' ? 0.6 : 0.35) : 0.0;
					else if (HAZARD_MOVES.has(move.id)) s = !side.foe.sideConditions[move.sideCondition || move.id] && side.foe.pokemonLeft >= 2 ? 0.4 : 0.0;
					cands.push({ c: `move ${j + 1}`, s });
					return;
				}
				const spread = ['allAdjacentFoes', 'allAdjacent'].includes(move.target);
				const tgts = doubles && !spread && ['normal', 'any', 'adjacentFoe'].includes(move.target) ? [1, 2] : [null];
				for (const t of tgts) {
					const tf = t ? side.foe.active[t - 1] : null;
					const targets = t ? [tf] : foes;
					let s = 0;
					for (const f of targets) {
						if (!f || f.fainted) continue;
						const fr = estFrac(b, me, f, move) * (spread && doubles && foes.length > 1 ? 0.75 : 1);
						const fh = f.hp / f.maxhp;
						s += Math.min(1, fr / Math.max(fh, 0.01)) * 0.6 + (fr >= fh ? 0.6 : 0) + (fr >= fh && move.priority > 0 ? 0.4 : 0);
						if (f.volatiles.substitute) s += (move.flags.sound || move.infiltrates || me.hasAbility('infiltrator')) ? 0.4 : (move.multihit ? 0.25 : -0.1);
						if (ANTI_SETUP.has(move.id) && ['atk', 'spa', 'spe'].reduce((t, k) => t + Math.max(0, f.boosts[k]), 0) >= 2) s += 0.8;
					}
					if (spread && move.target === 'allAdjacent' && doubles) {
						const ally = side.active[1 - si];
						if (ally && !ally.fainted) s -= 1.2 * estFrac(b, me, ally, move);
					}
					cands.push({ c: `move ${j + 1}` + (t ? ` ${t}` : ''), s });
				}
			});
			if (!a.trapped) {
				for (const x of bench) {
					if (usedSw.has(x.i)) continue;
					cands.push({ c: `switch ${x.i}`, s: matchup(x.p) - matchup(me) - 0.35, sw: x.i });
				}
			}
			const pick = pickSoft(cands) || { c: 'move 1' };
			if (pick.sw) usedSw.add(pick.sw);
			let c = pick.c;
			if (a.canMegaEvo && !megaDone && c.startsWith('move')) { c += ' mega'; megaDone = true; }
			parts.push(c);
		});
		return parts.join(', ');
	}

	// ---------------------------------------------------------------- 판세 평가 (선생님의 가치 판단)
	// 남은 포켓몬끼리의 대결을 빠르게 끝까지 따져 본다:
	//   나와 있는 둘이 서로 가장 센 기술을 주고받고(빠른 쪽 먼저), 쓰러지면 그 편은 남은 포켓몬 중
	//   지금 상대에게 가장 유리한 포켓몬을 낸다. 끝까지 남은 쪽의 남은 HP 로 승률을 정한다.
	function raceValue(b, meSide, oppSide) {
		// 사람이 머릿속으로 하는 '남은 포켓몬끼리의 대결'을 턴 단위로:
		//   공격 / 회복(회복기·희망사항) / 맹독 걸기 / 대타 세우기 중 그 상황에 맞는 것을 하고,
		//   턴 끝에 누적 데미지(맹독 1/16씩 증가·독·화상·모래)와 회복(먹다남은음식·검은진흙·포이즌힐)을 적용.
		//   새로 나오는 포켓몬은 장판(스텔스록·압정·끈적끈적네트)을 밟는다. 선공기는 스피드와 무관하게 먼저.
		const tr = !!b.field.getPseudoWeather('trickroom');
		const sand = b.field.isWeather('sandstorm');
		const has = (p, set) => p.moveSlots.some(ms => set.has(ms.id) && ms.pp > 0);
		const prioBest = (p, q) => {
			let best = 0;
			for (const ms of p.moveSlots) {
				const m = b.dex.moves.get(ms.id);
				if (m.priority > 0 && m.category !== 'Status') best = Math.max(best, estFrac(b, p, q, m));
			}
			return best;
		};
		const resid = (p) => {
			let r = 0;
			const mg = p.hasAbility('magicguard');
			if (p.hasItem('leftovers')) r += 1 / 16;
			if (p.hasItem('blacksludge')) r += p.hasType('Poison') ? 1 / 16 : -1 / 8;
			if (p.hasAbility('poisonheal') && (p.status === 'psn' || p.status === 'tox')) r += 1 / 8;
			else if (!mg) {
				if (p.status === 'psn') r -= 1 / 8;
				if (p.status === 'brn') r -= 1 / 16;
			}
			if (sand && !mg && !p.hasType('Rock') && !p.hasType('Ground') && !p.hasType('Steel') && !p.hasAbility('overcoat') && !p.hasItem('safetygoggles')) r -= 1 / 16;
			if (p.volatiles.leechseed && !mg) r -= 1 / 8;
			return r;
		};
		const mk = (p, act) => ({
			p, act, hp: p.hp / p.maxhp, sub: p.volatiles.substitute ? (p.volatiles.substitute.hp || 0) / p.maxhp : 0,
			status: p.status, tox: p.status === 'tox' ? Math.max(1, (p.statusState && p.statusState.stage) || 1) : 0,
			heal: has(p, RECOVER) ? 0.5 : (has(p, new Set(['wish'])) ? 0.4 : 0), heals: 0,
			canTox: has(p, new Set(['toxic'])), canSub: has(p, new Set(['substitute'])), resid: resid(p),
			spe: p.getStat('spe', false, true) * (p.status === 'par' ? 0.5 : 1), entered: act,
		});
		const live = (side) => side.pokemon.filter(p => !p.fainted && p.hp > 0).map(p => mk(p, p.isActive));
		let A = live(meSide), B = live(oppSide);
		if (!A.length || !B.length) return A.length ? 1 : (B.length ? 0 : 0.5);
		const enter = (x, side) => {
			if (x.entered) return;
			x.entered = true;
			const sc = side.sideConditions;
			if (x.p.hasItem('heavydutyboots') || x.p.hasAbility('magicguard')) return;
			const grounded = !x.p.hasType('Flying') && !x.p.hasAbility('levitate') && !x.p.hasItem('airballoon');
			if (sc.stealthrock) x.hp -= 0.125 * Math.pow(2, b.dex.getEffectiveness('Rock', x.p));
			if (sc.spikes && grounded) x.hp -= [0, 1 / 8, 1 / 6, 1 / 4][Math.min(3, sc.spikes.layers || 1)];
			if (sc.stickyweb && grounded) x.spe *= 2 / 3;
			if (sc.toxicspikes && grounded && !x.status && !x.p.hasType('Poison') && !x.p.hasType('Steel')) {
				x.status = (sc.toxicspikes.layers || 1) >= 2 ? 'tox' : 'psn';
				x.tox = x.status === 'tox' ? 1 : 0;
				x.resid = resid(x.p) - (x.status === 'psn' && !x.p.hasAbility('poisonheal') ? 1 / 8 : 0);
			}
		};
		const memo = new Map();
		const hit = (x, y) => {
			const k = x.p.position + x.p.side.id + '>' + y.p.position + y.p.side.id + x.p.species.id + y.p.species.id;
			if (!memo.has(k)) memo.set(k, Math.max(0.0001, bestVs(b, x.p, y.p)));
			return memo.get(k);
		};
		const first = (x, y) => tr ? x.spe < y.spe : x.spe > y.spe;
		const toxable = (y) => !y.status && !y.sub && !y.p.hasType('Steel') && !y.p.hasType('Poison') &&
			!y.p.hasAbility('immunity') && !y.p.hasAbility('purifyingsalt') && !y.p.hasAbility('comatose') && !y.p.hasAbility('magicbounce');
		const act = (x, y) => {
			if (x.hp <= 0) return;
			const dmg = hit(x, y);
			if (x.heal && x.heals < 8 && x.hp < 0.5 && hit(y, x) < x.heal - 0.05) { x.hp = Math.min(1, x.hp + x.heal); x.heals++; return; }
			if (x.canTox && toxable(y) && dmg < y.hp * 0.5) { y.status = 'tox'; y.tox = 1; return; }
			if (x.canSub && !x.sub && x.hp > 0.3 && hit(y, x) < 0.25 && dmg < y.hp * 0.5) { x.hp -= 0.25; x.sub = 0.25; return; }
			if (y.sub > 0) { y.sub = Math.max(0, y.sub - dmg); return; }
			// 기합의띠·옹골참: HP 가득에서 한 방이면 1 남김 (한 번)
			if (y.hp >= 0.999 && dmg >= y.hp && !y.endured && (y.p.hasItem('focussash') || y.p.hasAbility('sturdy'))) { y.hp = 0.003; y.endured = true; }
			else y.hp -= dmg;
			// 생명의구슬 반동 (HP 1 남은 포켓몬은 한 번 치고 쓰러짐)
			if (x.p.hasItem('lifeorb') && !x.p.hasAbility('magicguard') && !x.p.hasAbility('sheerforce')) x.hp -= 0.1;
		};
		const pickNext = (mine, foe) => {
			let best = null, bs = -1e9;
			for (const x of mine) {
				const s = hit(x, foe) / Math.max(foe.hp, 0.01) - hit(foe, x) / Math.max(x.hp, 0.01) + (first(x, foe) ? 0.3 : 0);
				if (s > bs) { bs = s; best = x; }
			}
			return best;
		};
		let a = A.find(x => x.act) || A[0];
		let o = B.find(x => x.act) || B[0];
		enter(a, meSide); enter(o, oppSide);
		let guard = 0;
		while (a && o && guard++ < 60) {
			// 선공기로 쓰러뜨릴 수 있으면 스피드와 무관하게 먼저
			const pa = prioBest(a.p, o.p) >= o.hp && !o.sub, po = prioBest(o.p, a.p) >= a.hp && !a.sub;
			let order = first(a, o) ? [[a, o], [o, a]] : [[o, a], [a, o]];
			if (pa && !po) order = [[a, o], [o, a]];
			else if (po && !pa) order = [[o, a], [a, o]];
			for (const [x, y] of order) act(x, y);
			for (const x of [a, o]) {
				if (x.hp <= 0) continue;
				x.hp = Math.min(1, x.hp + x.resid);
				if (x.status === 'tox' && !x.p.hasAbility('poisonheal') && !x.p.hasAbility('magicguard')) { x.hp -= x.tox / 16; x.tox = Math.min(15, x.tox + 1); }
			}
			if (a.hp <= 0) { A = A.filter(x => x !== a); a = A.length ? pickNext(A, o) : null; if (a) enter(a, meSide); }
			if (o.hp <= 0) { B = B.filter(x => x !== o); o = B.length ? pickNext(B, a || o) : null; if (o) enter(o, oppSide); }
		}
		const ma = A.reduce((s, x) => s + Math.max(0, x.hp), 0), mb = B.reduce((s, x) => s + Math.max(0, x.hp), 0);
		// 한쪽이 전멸하는 결말이면 남은 HP 가 적어도 승패는 확실하다 (HP 1 핫삼의 올킬도 승리)
		if (!B.length && A.length) return 0.88 + 0.1 * Math.min(1, ma);
		if (!A.length && B.length) return 0.12 - 0.1 * Math.min(1, mb);
		return 1 / (1 + Math.exp(-2.5 * (ma - mb)));
	}
	function evalSide(side) {
		let v = 0;
		for (const p of side.pokemon) {
			if (p.fainted || p.hp <= 0) continue;
			v += 0.35 + 0.65 * p.hp / p.maxhp;
			if (p.status && p.status !== 'fnt') v -= p.status === 'slp' ? 0.12 : (p.status === 'tox' ? 0.06 + 0.02 * ((p.statusState && p.statusState.stage) || 1) : 0.07);
			if (p.isActive) {
				for (const k of ['atk', 'spa', 'spe']) v += 0.03 * Math.max(0, p.boosts[k]) - 0.02 * Math.max(0, -p.boosts[k]);
				for (const k of ['def', 'spd']) v += 0.015 * p.boosts[k];
				if (p.volatiles.substitute) v += 0.06;
			}
		}
		return v;
	}
	function leafValue(b, me, mode) {
		if (b.ended) return b.winner === me ? 1 : (b.winner ? 0 : 0.5);
		const s = b.sides[0].id === me ? [b.sides[0], b.sides[1]] : [b.sides[1], b.sides[0]];
		const material = 1 / (1 + Math.exp(-2.2 * (evalSide(s[0]) - evalSide(s[1]))));
		if (mode === 'material') return material;
		const race = raceValue(b, s[0], s[1]);
		if (mode === 'race') return race;
		// 대결 결론이 확실하면(전멸) 남은 마리 수보다 그 결론을 믿는다
		const sure = race >= 0.88 || race <= 0.12;
		return sure ? 0.9 * race + 0.1 * material : 0.65 * race + 0.35 * material;
	}

	function mulberry(a) {
		return function () {
			a |= 0; a = a + 0x6D2B79F5 | 0;
			let t = Math.imul(a ^ a >>> 15, 1 | a);
			t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
			return ((t ^ t >>> 14) >>> 0) / 4294967296;
		};
	}

	function step(c, side, choice) {
		const ok = c.choose(side, choice);
		if (!ok) c.choose(side, 'default');
		c.log = []; c.sentLogPos = 0;
	}

	// ---------------------------------------------------------------- 탐색
	// sc: {gameType, mySide, oppSide, p1:{packed,state}, p2:{packed,state}, worlds:[{packed,state}], field, sides,
	//      megaUsed, turn, options:[...], rollouts, depth, temp, seed}
	function search(sc) {
		const me = sc.mySide;
		const rng = mulberry(sc.seed || 1);
		const bases = sc.worlds.map((w, k) => {
			try { return build(sc, w, [k + 1, 7, 11, (sc.seed || 1) % 65536]); } catch (e) { return null; }
		}).filter(Boolean);
		if (!bases.length) return { error: 'no world' };
		const depth = sc.depth || 3, R = sc.rollouts || 2, temp = sc.temp || 0.12;
		const budget = sc.budgetMs || 4000;   // 한 결정에 쓰는 최대 시간
		const tStart = Date.now();
		const out = [];
		// 같은 난수 비교: 모든 행동이 (세계 k, 굴림 r) 마다 같은 씨앗을 쓴다
		const seeds = [];
		for (let k = 0; k < bases.length; k++) {
			seeds.push([]);
			for (let r = 0; r < R; r++) seeds[k].push([1 + Math.floor(rng() * 65535), 1 + Math.floor(rng() * 65535), 1 + Math.floor(rng() * 65535), 1 + Math.floor(rng() * 65535)]);
		}
		for (const opt of sc.options) {
			let sum = 0, sq = 0, n = 0, bad = 0;
			const isEval = opt === null;
			for (let k = 0; k < bases.length; k++) {
				const base = bases[k];
				for (let r = 0; r < R; r++) {
					if (n > 0 && Date.now() - tStart > budget * (out.length + 1) / sc.options.length) break;
					const c = Battle.fromJSON(base.toJSON());
					c.resetRNG(seeds[k][r]);
					const prng = mulberry(seeds[k][r][0] * 65536 + seeds[k][r][1]);
					c.log = []; c.sentLogPos = 0;
					const startTurn = c.turn;
					if (!isEval) {
						const oppSide = c.sides.find(s => s.id !== me);
						const oppReq = oppSide.activeRequest;
						const ok = c.choose(me, opt);
						if (!ok) { bad++; continue; }
						if (oppReq && !oppReq.wait) step(c, oppSide.id, rolloutChoice(c, oppSide, prng, temp) || 'default');
					}
					let guard = 0;
					while (!c.ended && c.turn - startTurn < depth && guard++ < depth * 4) {
						let moved = false;
						for (const s of c.sides) {
							const rq = s.activeRequest;
							if (!rq || rq.wait) continue;
							step(c, s.id, rolloutChoice(c, s, prng, temp) || 'default');
							moved = true;
							if (c.ended) break;
						}
						if (!moved) break;
					}
					const v = leafValue(c, me, sc.leaf || 'mix');
					sum += v; sq += v * v; n++;
				}
			}
			const mean = n ? sum / n : 0;
			out.push({ value: n ? mean : null, n, sd: n > 1 ? Math.sqrt(Math.max(0, sq / n - mean * mean)) : 0, invalid: bad });
		}
		return { results: out, worlds: bases.length };
	}

	// ---------------------------------------------------------------- 선출 승률표
	// sc: {format, A:{packed}, B:{worlds:[packed...]}, ordersA:["123",...], ordersB:[...], games, temp, seed, budgetMs}
	// 각 칸: A 가 ordersA[i] 로, B 가 ordersB[j] 로 선출해 끝까지 대전한 A 의 승률.
	function playout(format, packedA, packedB, orderA, orderB, rng, temp, seed) {
		const b = new Battle({ formatid: format, seed });
		b.setPlayer('p1', { name: 'p1', team: packedA });
		b.setPlayer('p2', { name: 'p2', team: packedB });
		b.choose('p1', 'team ' + orderA);
		b.choose('p2', 'team ' + orderB);
		b.log = []; b.sentLogPos = 0;
		let guard = 0;
		while (!b.ended && guard++ < 300) {
			let moved = false;
			for (const s of b.sides) {
				const r = s.activeRequest;
				if (!r || r.wait) continue;
				const c = rolloutChoice(b, s, rng, temp) || 'default';
				if (!b.choose(s.id, c)) b.choose(s.id, 'default');
				b.log = []; b.sentLogPos = 0;
				moved = true;
				if (b.ended) break;
			}
			if (!moved) break;
		}
		return { winner: b.winner === 'p1' ? 1 : b.winner === 'p2' ? 0 : 0.5, turns: b.turn };
	}

	function matrix(sc) {
		const rng = mulberry(sc.seed || 1);
		const temp = sc.temp || 0.12;
		const G = sc.games || 4;
		const worlds = sc.B.worlds;
		const t0 = Date.now();
		const budget = sc.budgetMs || 60000;
		const M = [], N = [];
		let played = 0, turns = 0;
		for (let i = 0; i < sc.ordersA.length; i++) {
			M.push([]); N.push([]);
			for (let j = 0; j < sc.ordersB.length; j++) {
				let w = 0, n = 0;
				for (let g = 0; g < G; g++) {
					if (Date.now() - t0 > budget && n > 0) break;
					const world = worlds[(i * 7 + j * 3 + g) % worlds.length];
					const seed = [1 + ((i * 131 + j * 17 + g * 7 + (sc.seed || 1)) % 65000), 2 + g, 3 + i, 4 + j];
					try {
						const r = playout(sc.format, sc.A.packed, world, sc.ordersA[i], sc.ordersB[j], rng, temp, seed);
						w += r.winner; n++; turns += r.turns;
					} catch (e) { /* 깨진 판은 건너뜀 */ }
				}
				played += n;
				M[i].push(n ? w / n : 0.5); N[i].push(n);
			}
		}
		return { M, N, played, avgTurns: played ? turns / played : 0, ms: Date.now() - t0 };
	}

	return { search, build, rolloutChoice, matrix, playout, leafValue, raceValue };
};

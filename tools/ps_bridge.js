#!/usr/bin/env node
/*
 * 쇼다운 시뮬레이터 다리 — 파이썬이 표준입출력(JSON 한 줄)으로 대전을 조종한다.
 *
 *   node tools/ps_bridge.js <쇼다운 폴더>
 *
 * 입력 (한 줄에 JSON 하나)
 *   {"op":"formats"}
 *   {"op":"start","id":"g1","format":"gen9championsbssregmb","seed":[1,2,3,4],"p1":"<packed>","p2":"<packed>"}
 *   {"op":"choose","id":"g1","side":"p1","choice":"move 1"}
 *   {"op":"end","id":"g1"}
 * 출력 (요청마다 한 줄)
 *   {"id":"g1","log":{"p1":[...],"p2":[...],"omni":[...],"spec":[...]},"requests":{"p1":{...}|null,"p2":...},
 *    "ended":false,"winner":null,"turn":3}
 *
 * 로그는 쇼다운 프로토콜 그대로이며, |split| 비밀/공개 줄을 각 플레이어 시점으로 나눠 준다.
 * (같은 해석기로 실제 쇼다운 리플레이도 읽을 수 있게 하기 위함)
 */
'use strict';
const path = require('path');
const readline = require('readline');

const psDir = path.resolve(process.argv[2] || process.env.PS_DIR || 'build/pokemon-showdown');
const { Battle, Dex, Teams } = require(path.join(psDir, 'dist', 'sim'));

const battles = new Map();
const searcher = require('./ps_search.js')(Battle);

function splitLog(lines) {
	const out = { p1: [], p2: [], omni: [], spec: [] };
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const m = /^\|split\|(p[1-4])$/.exec(line);
		if (m) {
			const secret = lines[i + 1], pub = lines[i + 2];
			for (const s of ['p1', 'p2']) out[s].push(s === m[1] ? secret : pub);
			out.omni.push(secret);
			out.spec.push(pub);   // 관전자 시점 (= 쇼다운 리플레이와 같은 정보)
			i += 2;
			continue;
		}
		out.p1.push(line); out.p2.push(line); out.omni.push(line); out.spec.push(line);
	}
	return out;
}

function makeBattle(msg) {
	const st = { log: [], requests: { p1: null, p2: null }, ended: false, winner: null };
	const battle = new Battle({
		formatid: msg.format, seed: msg.seed || undefined,
		send: (type, data) => {
			if (type === 'update') {
				st.log.push(...data);
			} else if (type === 'sideupdate') {
				const [side, ...rest] = String(data).split('\n');
				const body = rest.join('\n');
				if (body.startsWith('|request|')) {
					st.requests[side] = JSON.parse(body.slice('|request|'.length));
				} else if (body.startsWith('|error|')) {
					st.lastError = st.lastError || {};
					st.lastError[side] = body.slice('|error|'.length);
				}
			} else if (type === 'end') {
				st.ended = true;
			}
		},
	});
	st.battle = battle;
	battle.setPlayer('p1', { name: 'p1', team: msg.p1 });
	battle.setPlayer('p2', { name: 'p2', team: msg.p2 });
	return st;
}

function respond(id, st, extra) {
	st.battle.sendUpdates();
	const log = splitLog(st.log);
	st.log = [];
	const b = st.battle;
	const reqs = {};
	for (const s of ['p1', 'p2']) {
		const r = st.requests[s];
		reqs[s] = r && !r.wait ? r : null;
	}
	const out = {
		id, log, requests: reqs, ended: b.ended, winner: b.winner || null, turn: b.turn,
		errors: st.lastError || null, ...extra,
	};
	st.lastError = null;
	return out;
}

function handle(msg) {
	switch (msg.op) {
	case 'formats':
		return {
			formats: Dex.formats.all().filter(f => f.mod && f.mod.startsWith('champions') && f.effectType === 'Format')
				.map(f => ({ id: f.id, name: f.name, mod: f.mod, gameType: f.gameType || 'singles' })),
		};
	case 'start': {
		const st = makeBattle(msg);
		battles.set(msg.id, st);
		return respond(msg.id, st);
	}
	case 'choose': {
		const st = battles.get(msg.id);
		if (!st) return { id: msg.id, error: 'no battle' };
		st.requests[msg.side] = null;
		const ok = st.battle.choose(msg.side, msg.choice);
		if (!ok) {
			// 잘못된 선택: 요청을 다시 돌려준다
			st.requests[msg.side] = st.battle.sides[msg.side === 'p1' ? 0 : 1].activeRequest;
		}
		const r = respond(msg.id, st, { accepted: !!ok });
		if (st.battle.ended) battles.delete(msg.id);
		return r;
	}
	case 'end':
		battles.delete(msg.id);
		return { id: msg.id, ended: true };
	case 'search': {
		const t0 = Date.now();
		if (process.env.SEARCH_DUMP) require('fs').writeFileSync(process.env.SEARCH_DUMP, JSON.stringify(msg));
		const r = searcher.search(msg);
		r.ms = Date.now() - t0;
		return r;
	}
	case 'matrix':
		return searcher.matrix(msg);
	case 'pack':
		return { packed: Teams.pack(msg.team) };
	default:
		return { error: 'unknown op ' + msg.op };
	}
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
	if (!line.trim()) return;
	let out;
	try {
		out = handle(JSON.parse(line));
	} catch (e) {
		out = { error: String(e && e.stack || e) };
	}
	process.stdout.write(JSON.stringify(out) + '\n');
});

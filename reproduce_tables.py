"""Regenerate the paper's tables and figures from the released per-response data.

Reads data/tasks/<set>.jsonl and data/responses.jsonl.gz (one line per model x task x representation; the last
successful response of each request) and writes out/tables/*.csv|tex and out/figures/*.pdf. No model calls.
Needs Python with numpy, pandas and matplotlib.  Usage: python reproduce_tables.py
"""
from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
REPS = ['table_nm', 'table_um', 'table_sci', 'table_ev', 'plot_nm', 'plot_ev']
TABLES = REPS[:4]
BASE = ['d3-nano', 'jev', 'luna-openrouter', 'luna-chat', 'luna-chat-low']
QWEN = 'qwen35-2b'
LABEL = dict(zip(BASE + [QWEN], ['d3-nano', 'Jev', 'Luna Decisions', 'Luna gen. none', 'Luna gen. low',
                                 'Qwen3.5-2B base']))
ARMS = ['ftA', 'ftAc', 'ftBm', 'ftBmc', 'ftB', 'ftC']
NEW_ARMS = ['ftD', 'ftCc', 'ftDc']
COLORS = dict(zip(ARMS + NEW_ARMS, ['#B7B7B7', '#D89A43', '#7A9EAC', '#397A99', '#7956A0', '#CF5140',
                                    '#3E8E5B', '#8C2D2D', '#9BC48F']))
SETS = ['main_v1', 'gold_v1', 'gold_v1t', 'suff_v3', 'gold_suff_v3']
E_SETS = ['main_v1', 'gold_v1', 'suff_v3', 'gold_suff_v3']
DS = {'main_v1': 'MOCLIP', 'gold_v1': 'Gold', 'suff_v3': 'MOCLIP', 'gold_suff_v3': 'Gold', 'gold_v1t': 'Gold T'}
for sub in ['figures', 'tables']:
    (OUT / sub).mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'savefig.bbox': 'tight'})


def tasks(name):
    p = DATA / 'tasks' / f'{name}.jsonl'
    return {d['task_id']: d for d in map(json.loads, p.read_text().splitlines()) if '_meta' not in d}


def arm_models(arm):
    return [f'd3-nano-{arm}' if s == 0 and arm in ['ftA', 'ftB'] else f'd3-nano-{arm}_s{s}' for s in range(3)]


def allowed(t, rep):
    only = t['params'].get('reps')
    return only is None or rep in only or (rep == 'table_ev_desc' and 'table_ev' in only)


RESP = {}
with gzip.open(DATA / 'responses.jsonl.gz', 'rt', encoding='utf-8') as fh:
    for line in fh:
        d = json.loads(line)
        RESP.setdefault((d['dataset'], d['batch']), []).append(d)


def load(name, sub='raw', strict=True):
    ts = tasks(name)
    out = []
    for d in RESP.get((name, sub), []):
        assert d['task_id'] in ts and d['answer'] == ts[d['task_id']]['answer']
        d = dict(d)
        for k in ('pick_first', 'letter_mass'):
            d[k] = np.nan if d[k] is None else d[k]
        out.append(d)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    for model, g in df.groupby('model'):
        reps = sorted(set(g.rep))
        expected = {(tid, rep) for tid, t in ts.items() for rep in reps if allowed(t, rep)}
        got = set(zip(g.task_id, g.rep))
        if strict:
            assert got == expected, (name, sub, model, 'coverage', len(got), len(expected))
        elif got != expected:
            print(f'[partial] {name}/{sub}/{model}: {len(got)}/{len(expected)} requests')
            df = df[df.model != model]
    if 'oracle' in set(df.model):
        assert df[df.model == 'oracle'].correct.eq(1).all()
    return df


def boot(g, col, by='item'):
    z = g.groupby(by)[col].agg(['sum', 'count'])
    ix = np.random.default_rng(0).integers(0, len(z), size=(2000, len(z)))
    vals = z['sum'].to_numpy()[ix].sum(1) / z['count'].to_numpy()[ix].sum(1)
    return g[col].mean(), *np.quantile(vals, [.025, .975])


def cell(v, signed=False, ci=True):
    fmt = '{:+.0f}' if signed else '{:.0f}'
    if not ci:
        return fmt.format(100 * v[0])
    return '{} [{}, {}]'.format(*(fmt.format(100 * x) for x in v))


def write_rows(name, rows):
    (OUT / 'tables' / f'{name}.tex').write_text('\n'.join(' & '.join(map(str, r)) + r' \\' for r in rows) + '\n')


D = {n: load(n) for n in SETS}
E = {n: load(n, 'raw_102e', strict=False) for n in E_SETS if (n, 'raw_102e') in RESP}
HAVE = {n: set(E[n].model) if n in E and not E[n].empty else set() for n in E_SETS}
print('102e models present:', {n: sorted(v) for n, v in HAVE.items()})


def qwen_rows(name):
    """Qwen3.5-2B (102e batch, six standard representations) appended to a historical frame."""
    if QWEN not in HAVE.get(name, set()):
        return D[name]
    q = E[name][(E[name].model == QWEN) & E[name].rep.isin(REPS)]
    return pd.concat([D[name], q], ignore_index=True)


A = {}
for n in SETS:
    A[n] = D[n]
    if n in E_SETS:
        A[n] = qwen_rows(n)
        extra = [E[n][E[n].model == m] for a in NEW_ARMS for m in arm_models(a) if m in HAVE[n]]
        if extra:
            A[n] = pd.concat([A[n]] + extra, ignore_index=True)
MODELS = BASE + ([QWEN] if QWEN in HAVE.get('main_v1', set()) else [])

acc = pd.concat([g.groupby(['dataset', 'model', 'type', 'rep', 'core']).correct.agg(['mean', 'count']).reset_index()
                 for g in A.values()])
acc.to_csv(OUT / 'tables' / 'accuracy.csv', index=False)

# ---------- Table: core accuracy by modality ----------
rows = []
for model in MODELS:
    r = [LABEL[model]]
    for name in SETS[:2]:
        df = A[name]; g = df[(df.model == model) & df.core]
        for kind in ['table', 'plot']:
            z = g[g.rep.str.startswith(kind)]
            r.append('--' if z.empty else cell(boot(z, 'correct')))
    rows.append(r)
write_rows('accuracy', rows)

# ---------- Table: paired representation effects against the nm table ----------
eff_rows = []
for name in SETS[:2]:
    df = A[name]
    for model in MODELS:
        g = df[(df.model == model) & df.core]
        p = g.pivot(index=['item', 'task_id'], columns='rep', values='correct')
        for rep in REPS[1:]:
            if rep not in p:
                continue
            v = (p[rep] - p['table_nm']).rename('d').reset_index()
            est = boot(v, 'd')
            eff_rows.append(dict(dataset=name, model=model, rep=rep, diff=est[0], lo=est[1], hi=est[2]))
eff = pd.DataFrame(eff_rows); eff.to_csv(OUT / 'tables' / 'rep_effects.csv', index=False)
rows = []
for name in SETS[:2]:
    for model in MODELS:
        r = [DS[name] if model == MODELS[0] else '', LABEL[model]]
        for rep in REPS[1:]:
            z = eff[(eff.dataset == name) & (eff.model == model) & (eff.rep == rep)]
            r.append('--' if z.empty else cell(z.iloc[0][['diff', 'lo', 'hi']], True))
        rows.append(r)
write_rows('rep_effects', rows)

# ---------- Table: agreement decomposed, with an independence baseline ----------
cons_rows = []
for name in SETS[:2]:
    df = A[name]
    for model in MODELS:
        g = df[(df.model == model) & df.core]
        for scope, reps in [('core_tables', TABLES), ('core_all', REPS)]:
            x = g[g.rep.isin(reps)]
            if set(x.rep) != set(reps):
                continue
            z = x.groupby(['item', 'task_id']).agg(n=('choice', 'nunique'), allc=('correct', 'min')).reset_index()
            z['agree'] = z.n.eq(1).astype(int); z['agree_correct'] = (z.n.eq(1) & z.allc.eq(1)).astype(int)
            z['agree_wrong'] = z.agree - z.agree_correct
            accs = x.groupby(['type', 'rep']).correct.mean().unstack()
            ntype = x.groupby('type').task_id.nunique()
            indep = float((accs.prod(axis=1) * ntype).sum() / ntype.sum())
            rec = dict(dataset=name, model=model, scope=scope, n=len(z), independence_all_correct=indep)
            for col in ['agree', 'agree_correct', 'agree_wrong']:
                rec[col], rec[col + '_lo'], rec[col + '_hi'] = boot(z, col)
            cons_rows.append(rec)
cons = pd.DataFrame(cons_rows); cons.to_csv(OUT / 'tables' / 'consistency.csv', index=False)
rows = []
for model in MODELS:
    r = [LABEL[model]]
    for name in SETS[:2]:
        t4 = cons[(cons.dataset == name) & (cons.model == model) & (cons.scope == 'core_tables')]
        a6 = cons[(cons.dataset == name) & (cons.model == model) & (cons.scope == 'core_all')]
        r.append(cell(t4.iloc[0][['agree', 'agree_lo', 'agree_hi']]))
        if a6.empty:
            r += ['--', '--', '--']
        else:
            z = a6.iloc[0]
            r += [cell(z[['agree', 'agree_lo', 'agree_hi']]), f'{100 * z.agree_wrong:.0f}', f'{100 * z.independence_all_correct:.0f}']
    rows.append(r)
write_rows('consistency', rows)

# Where inconsistency arises: share of inconsistent core variants by task type (all six representations).
inc_rows = []
for name in SETS[:2]:
    df = A[name]
    for model in MODELS:
        g = df[(df.model == model) & df.core]
        if set(g.rep) != set(REPS):
            continue
        z = g.groupby(['type', 'task_id']).choice.nunique().gt(1).groupby('type').agg(['sum', 'count'])
        for typ, v in z.iterrows():
            inc_rows.append(dict(dataset=name, model=model, type=typ, inconsistent=int(v['sum']), variants=int(v['count'])))
pd.DataFrame(inc_rows).to_csv(OUT / 'tables' / 'inconsistency_by_type.csv', index=False)

# ---------- Table: interface and reasoning (core variants) ----------
rows = []
for name in SETS[:2]:
    df = A[name]; df = df[df.core]
    piv = df[df.model.isin(BASE[2:])].pivot(index=['item', 'task_id', 'rep'], columns='model', values='correct').reset_index()
    r = [DS[name]]
    for model in BASE[3:]:
        z = piv.assign(diff=piv[model] - piv['luna-openrouter']); r.append(cell(boot(z, 'diff'), True))
    for model in BASE[2:]:
        r.append(cell(boot(df[df.model == model], 'hcw')))
    rows.append(r)
write_rows('interface', rows)

# Evidence-order preference: share choosing the first-listed spectrum on balanced core compare tables.
pos_rows = []
for name in SETS[:2]:
    df = A[name]
    for model in MODELS + ['d3-nano-pa']:
        g = df[(df.model == model) & df.core & (df.type == 'compare') & df.rep.str.startswith('table')]
        if g.empty:
            continue
        est = boot(g.assign(pf=g.pick_first.astype(float)), 'pf')
        pos_rows.append(dict(dataset=name, model=model, first=est[0], lo=est[1], hi=est[2], truth_first=(g.answer == 'A').mean()))
pd.DataFrame(pos_rows).to_csv(OUT / 'tables' / 'position.csv', index=False)

# ---------- Table: wording effects, full matrix ----------
ev_rows = []
for name in SETS[:2]:
    df = A[name]
    sel = df[df.model.isin(MODELS) & df.type.isin(['compare', 'threshold']) & df.rep.isin(['table_ev', 'plot_ev'])]
    for (model, typ, rep), g in sel.groupby(['model', 'type', 'rep']):
        p = g.pivot(index=['item', 'order', 'direction'], columns='phrasing', values='correct')
        v = (p['axis'] - p['wavelength']).rename('diff').reset_index(); est = boot(v, 'diff')
        ev_rows.append(dict(dataset=name, model=model, type=typ, rep=rep, wavelength=p.wavelength.mean(),
                            energy=p.axis.mean(), diff=est[0], lo=est[1], hi=est[2]))
ev = pd.DataFrame(ev_rows); ev.to_csv(OUT / 'tables' / 'energy_wording.csv', index=False)
rows = []
for model in MODELS:
    r = [LABEL[model]]
    for name, typ in [('main_v1', 'compare'), ('gold_v1', 'compare'), ('gold_v1', 'threshold')]:
        for rep in ['table_ev', 'plot_ev']:
            z = ev[(ev.dataset == name) & (ev.model == model) & (ev.type == typ) & (ev.rep == rep)]
            r.append('--' if z.empty else
                     f'{100 * z.iloc[0].wavelength:.0f}$\\to${100 * z.iloc[0].energy:.0f} ({cell(z.iloc[0][["diff", "lo", "hi"]], True)})')
    rows.append(r)
write_rows('energy_wording', rows)

# ---------- Scope sets: discrimination, outside-cut change, full-view accuracy ----------
suff_rows = []
for name in SETS[-2:]:
    df = A[name]
    for (model, kind), g in df.groupby(['model', df.rep.str.split('_').str[0]]):
        p = g.pivot(index=['item', 'rep'], columns='cut', values=['undet', 'correct'])
        u = (p['undet']['in'] - p['undet']['out']).rename('v').reset_index()
        a = (p['correct']['out'] - p['correct']['none']).rename('v').reset_index()
        f = p['correct']['none'].rename('v').reset_index()
        uv, av, fv = boot(u, 'v'), boot(a, 'v'), boot(f, 'v')
        suff_rows.append(dict(dataset=name, model=model, kind=kind, discrimination=uv[0], d_lo=uv[1], d_hi=uv[2],
                              outside_delta=av[0], o_lo=av[1], o_hi=av[2], full=fv[0], f_lo=fv[1], f_hi=fv[2]))
suff = pd.DataFrame(suff_rows); suff.to_csv(OUT / 'tables' / 'sufficiency.csv', index=False)


def get_suff(name, model, kind):
    z = suff[(suff.dataset == name) & (suff.model == model) & (suff.kind == kind)]
    return None if z.empty else z.iloc[0]


rows = []
for model in MODELS + arm_models('ftC'):
    r = [LABEL.get(model, 'ftC seed ' + model[-1])]
    for name, kind in [('suff_v3', 'table'), ('suff_v3', 'plot'), ('gold_suff_v3', 'table')]:
        z = get_suff(name, model, kind)
        r.extend(['--'] * 3 if z is None else [f'{100 * z.full:.0f}', f'{100 * z.discrimination:+.0f}', f'{100 * z.outside_delta:+.0f}'])
    rows.append(r)
write_rows('sufficiency', rows)
rows = []
for model in MODELS:
    for name in SETS[-2:]:
        for kind in ['table', 'plot']:
            z = get_suff(name, model, kind)
            if z is None:
                continue
            rows.append([LABEL[model], DS[name], kind, cell(z[['full', 'f_lo', 'f_hi']]),
                         cell(z[['discrimination', 'd_lo', 'd_hi']], True), cell(z[['outside_delta', 'o_lo', 'o_hi']], True)])
write_rows('scope_base_intervals', rows)

# ---------- Training arms (core convention) ----------
present_new = [a for a in NEW_ARMS if all(m in HAVE.get('suff_v3', set()) and m in HAVE.get('main_v1', set())
                                          and m in HAVE.get('gold_v1', set()) and m in HAVE.get('gold_suff_v3', set())
                                          for m in arm_models(a))]
ALL_ARMS = ARMS + present_new
rows = []
for arm in ALL_ARMS:
    for seed, model in enumerate(arm_models(arm)):
        r = [arm, str(seed)]
        for name, typ, kind in [('main_v1', 'compare', 'table'), ('main_v1', 'compare', 'plot'), ('main_v1', 'spec', 'table'),
                                ('gold_v1', 'threshold', 'table')]:
            g = A[name]; z = g[(g.model == model) & (g.type == typ) & g.rep.str.startswith(kind) & g.core]
            r.append(f'{100 * z.correct.mean():.0f}')
        for name in SETS[-2:]:
            z = get_suff(name, model, 'table'); r.extend([f'{100 * z.full:.0f}', f'{100 * z.discrimination:+.0f}', f'{100 * z.outside_delta:+.0f}'])
        rows.append(r)
write_rows('all_seeds', rows)
rows = []
for arm in ALL_ARMS:
    for seed, model in enumerate(arm_models(arm)):
        for name in SETS[-2:]:
            for kind in ['table', 'plot']:
                z = get_suff(name, model, kind)
                rows.append([arm, seed, DS[name], kind, cell(z[['full', 'f_lo', 'f_hi']]),
                             cell(z[['discrimination', 'd_lo', 'd_hi']], True), cell(z[['outside_delta', 'o_lo', 'o_hi']], True)])
write_rows('scope_intervals', rows)

# Training logs: selected epoch / final development accuracy for every adapter (released summary).
(OUT / 'tables' / 'training_logs.csv').write_bytes((DATA / 'training_logs.csv').read_bytes())

# Rule probe on gold threshold tables (core): agreement with the reversed, minimum-seeking rule where it differs.
tg = tasks('gold_v1')
inv = {}
for tid, t in tg.items():
    if t['type'] == 'threshold':
        sp = t['spectra'][0]; w = np.array(sp['wl_nm']); v = np.array(sp['t'])
        inv[tid] = 'longer' if w[v.argmin()] > t['params']['x_nm'] else 'shorter'
rp_rows = []
g = A['gold_v1']; g = g[g.core & (g.type == 'threshold') & g.rep.str.startswith('table')].copy()
g['inv'] = g.task_id.map(inv)
for arm in ALL_ARMS:
    for seed, model in enumerate(arm_models(arm)):
        z = g[(g.model == model) & (g.inv != g.answer)]
        rp_rows.append(dict(arm=arm, seed=seed, n=len(z), agree_inverse=float((z.choice == z.inv).mean())))
pd.DataFrame(rp_rows).to_csv(OUT / 'tables' / 'rule_probe.csv', index=False)

# gold_v1t auxiliary transmission view (core convention).
rows = []
g = D['gold_v1t']
for model in ['d3-nano'] + [m for a in ['ftA', 'ftBm', 'ftB'] for m in arm_models(a)]:
    z = g[(g.model == model) & g.core & g.rep.str.startswith('table')]
    rows.append(dict(model=model, threshold=z[z.type == 'threshold'].correct.mean(), compare=z[z.type == 'compare'].correct.mean()))
pd.DataFrame(rows).to_csv(OUT / 'tables' / 'gold_transmission.csv', index=False)

# ---------- 102e E1: eV row order (same batch only) ----------
e1_rows = []
for name in E_SETS:
    if name not in E or E[name].empty:
        continue
    df = E[name]
    for model in sorted(set(df.model) - {'oracle'}):
        g = df[(df.model == model) & df.rep.isin(['table_ev', 'table_ev_desc'])]
        if set(g.rep) != {'table_ev', 'table_ev_desc'}:
            continue
        p = g.pivot(index=['item', 'task_id'], columns='rep', values='correct')
        meta = g.drop_duplicates('task_id').set_index('task_id')
        p = p.join(meta[['type', 'phrasing', 'trunc_kind', 'truncated', 'cut']], on='task_id')
        u = g.pivot(index=['item', 'task_id'], columns='rep', values='undet').join(meta[['type', 'trunc_kind', 'truncated', 'cut']], on='task_id')
        groups = []
        for typ in sorted(set(p.type)):
            groups.append((typ, 'accuracy', p[p.type == typ]))
        if name == 'main_v1':
            groups.append(('compare-wavelength', 'accuracy', p[(p.type == 'compare') & (p.phrasing == 'wavelength')]))
            groups.append(('compare-energy', 'accuracy', p[(p.type == 'compare') & (p.phrasing == 'axis')]))
            mid = u[(u.type == 'sufficiency') & (u.truncated == True) & (u.trunc_kind == 'hide_middle')]  # noqa: E712
            groups.append(('middle-gap undetermined', 'undet', mid))
        if name == 'gold_v1':
            mid = u[(u.type == 'sufficiency') & (u.truncated == True) & (u.trunc_kind == 'hide_middle')]  # noqa: E712
            groups.append(('middle-gap undetermined', 'undet', mid))
        for label, metric, x in groups:
            if x.empty:
                continue
            v = (x['table_ev_desc'] - x['table_ev']).rename('d').reset_index()
            est = boot(v, 'd')
            e1_rows.append(dict(dataset=name, model=model, group=label, metric=metric, asc=x['table_ev'].mean(),
                                desc=x['table_ev_desc'].mean(), diff=est[0], lo=est[1], hi=est[2], n=len(x)))
e1 = pd.DataFrame(e1_rows)
if not e1.empty:
    e1.to_csv(OUT / 'tables' / 'ev_order.csv', index=False)
    # Scope discrimination difference (desc - asc) with a paired bootstrap over instances.
    sc_rows = []
    for name in SETS[-2:]:
        if name not in E:
            continue
        df = E[name]
        for model in sorted(set(df.model) - {'oracle'}):
            g = df[(df.model == model) & df.rep.isin(['table_ev', 'table_ev_desc'])]
            if set(g.rep) != {'table_ev', 'table_ev_desc'}:
                continue
            pp = g.pivot(index=['item', 'rep'], columns='cut', values=['undet', 'correct'])
            dsc = (pp['undet']['in'] - pp['undet']['out']).unstack('rep')
            out_ = (pp['correct']['out'] - pp['correct']['none']).unstack('rep')
            v = (dsc['table_ev_desc'] - dsc['table_ev']).rename('d').reset_index(); est = boot(v, 'd')
            sc_rows.append(dict(dataset=name, model=model, d_asc=dsc['table_ev'].mean(), d_desc=dsc['table_ev_desc'].mean(),
                                d_diff=est[0], d_lo=est[1], d_hi=est[2], out_asc=out_['table_ev'].mean(), out_desc=out_['table_ev_desc'].mean()))
    pd.DataFrame(sc_rows).to_csv(OUT / 'tables' / 'ev_order_scope.csv', index=False)
    order = [m for m in BASE + [QWEN] if any(e1.model == m)]
    rows = []
    sc = pd.DataFrame(sc_rows)

    def fmt(z, a, b, d, lo, hi, signed=False):
        f = '{:+.0f}' if signed else '{:.0f}'
        return f'{f.format(100 * z[a])}$\\to${f.format(100 * z[b])} ({cell(z[[d, lo, hi]], True)})'

    for model in order:
        r = [LABEL[model]]
        for name, grp in [('main_v1', 'compare-wavelength'), ('main_v1', 'compare-energy'), ('main_v1', 'middle-gap undetermined')]:
            z = e1[(e1.dataset == name) & (e1.model == model) & (e1.group == grp)]
            r.append('--' if z.empty else fmt(z.iloc[0], 'asc', 'desc', 'diff', 'lo', 'hi'))
        z = sc[(sc.dataset == 'suff_v3') & (sc.model == model)]
        r.append('--' if z.empty else fmt(z.iloc[0], 'd_asc', 'd_desc', 'd_diff', 'd_lo', 'd_hi', True))
        z = e1[(e1.dataset == 'gold_v1') & (e1.model == model) & (e1.group == 'compare')]
        r.append('--' if z.empty else fmt(z.iloc[0], 'asc', 'desc', 'diff', 'lo', 'hi'))
        z = sc[(sc.dataset == 'gold_suff_v3') & (sc.model == model)]
        r.append('--' if z.empty else fmt(z.iloc[0], 'd_asc', 'd_desc', 'd_diff', 'd_lo', 'd_hi', True))
        rows.append(r)
    write_rows('ev_order', rows)
    # API drift check: same-day ascending rerun vs historical ascending responses.
    drift = []
    for name in E_SETS:
        if name not in E:
            continue
        for model in sorted(set(E[name].model) & set(BASE)):
            new = E[name][(E[name].model == model) & (E[name].rep == 'table_ev')].set_index('task_id').choice
            old = D[name][(D[name].model == model) & (D[name].rep == 'table_ev')].set_index('task_id').choice
            common = new.index.intersection(old.index)
            drift.append(dict(dataset=name, model=model, n=len(common), same_choice=float((new[common] == old[common]).mean()),
                              acc_old=float(D[name][(D[name].model == model) & (D[name].rep == 'table_ev')].correct.mean()),
                              acc_new=float(E[name][(E[name].model == model) & (E[name].rep == 'table_ev')].correct.mean())))
    pd.DataFrame(drift).to_csv(OUT / 'tables' / 'rerun_drift.csv', index=False)

# ---------- 102e E2: same-backbone comparison ----------
if QWEN in MODELS:
    qrows = []
    for name in SETS[:2]:
        df = A[name]
        for model in ['d3-nano', QWEN]:
            g = df[(df.model == model) & df.core]
            qrows.append(dict(dataset=name, model=model, table=g[g.rep.str.startswith('table')].correct.mean(),
                              plot=g[g.rep.str.startswith('plot')].correct.mean(),
                              letter_mass=g.letter_mass.mean() if model == QWEN else np.nan))
    pd.DataFrame(qrows).to_csv(OUT / 'tables' / 'qwen_base.csv', index=False)

tq = tasks('suff_v3')

# ---------- Figures ----------
# Figure 1: real supplied arrays; explanatory shading/labels are added only for the paper.
target = next(t for t in tq.values() if t['params'].get('cut') == 'in' and t['params']['trunc_kind'] == 'hide_middle')
item = target['params']['item']; variants = {t['params']['cut']: t for t in tq.values() if t['params']['item'] == item}
fig, axes = plt.subplots(2, 3, figsize=(9, 4.5), sharey=True, layout='constrained')
for row in range(2):
    for col, (cut, title) in enumerate([('none', 'Full evidence'), ('out', 'Remove outside query'), ('in', 'Remove inside query')]):
        t = variants[cut]; s = t['spectra'][0]; w = np.array(s['wl_nm']); y = np.array(s['t']); q = t['params']['query_nm']; ax = axes[row, col]
        gaps = np.flatnonzero(np.diff(w) > 15)
        w = np.insert(w, gaps + 1, np.nan); y = np.insert(y, gaps + 1, np.nan)
        x = w if row == 0 else 1239.841984 / w; bounds = q if row == 0 else sorted(1239.841984 / np.array(q))
        ax.axvspan(*bounds, color='#E0EAF1', alpha=.8); ax.plot(x, y, 'o-', color='#245B78', ms=3, lw=1.5)
        ax.set_ylim(0, 1); ax.set_xlim((450, 730) if row == 0 else (1239.841984 / 730, 1239.841984 / 450))
        ax.set_xlabel('Wavelength (nm)' if row == 0 else 'Photon energy (eV)'); ax.grid(alpha=.15)
        if col == 0: ax.set_ylabel('Transmission')
        if row == 0: ax.set_title(title, fontsize=10)
        if row == 1: ax.text(.5, -.37, 'Answer preserved' if cut != 'in' else 'Cannot be determined', transform=ax.transAxes,
                             ha='center', color='#245B78' if cut != 'in' else '#B04535')
fig.savefig(OUT / 'figures' / 'scope_example.pdf'); plt.close(fig)

# Figure 2: core wavelength-worded compare tasks, all six representations.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.1 + 0.3 * (len(MODELS) - 5)), layout='constrained')
for ax, name, title in zip(axes, SETS[:2], ['MOCLIP: compare', 'Gold: compare']):
    df = A[name]; g = df[df.type.eq('compare') & df.core & df.model.isin(MODELS)]
    mat = g.pivot_table(index='model', columns='rep', values='correct').reindex(index=MODELS, columns=REPS) * 100
    ax.imshow(mat, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    for i in range(len(MODELS)):
        for j in range(6):
            v = mat.iloc[i, j]; ax.text(j, i, '—' if pd.isna(v) else f'{v:.0f}', ha='center', va='center', color='white' if v > 65 else '#172C3B')
    ax.set_xticks(range(6), ['nm', 'µm', 'sci.', 'eV', 'nm plot', 'eV plot'], rotation=30, ha='right')
    ax.set_yticks(range(len(MODELS)), [LABEL[m] for m in MODELS] if name == 'main_v1' else []); ax.set_title(title)
fig.savefig(OUT / 'figures' / 'compare_representation.pdf'); plt.close(fig)

# Figure 3: every optimization seed, core compare variants only.
fig, axes = plt.subplots(1, 3, figsize=(9, 3.0), sharey=True, layout='constrained')
df = A['main_v1']; cmpc = df[(df.type == 'compare') & df.core]
for ax, arm, title in zip(axes, ['ftAc', 'ftBmc', 'ftB'], ['nm only, selected epoch\n632 examples', 'Four tables, selected epoch\n632 examples',
                                                            'Four tables, 2 epochs\n2,956 examples']):
    base = cmpc[cmpc.model == 'd3-nano'].groupby('rep').correct.mean().reindex(REPS)
    ax.plot(range(6), 100 * base, '--', color='#777777', label='Base')
    for seed, model in enumerate(arm_models(arm)):
        vals = cmpc[cmpc.model == model].groupby('rep').correct.mean().reindex(REPS)
        ax.plot(np.arange(6) + (seed - 1) * .07, 100 * vals, marker=['o', 's', '^'][seed], lw=.9, ms=4, color=COLORS[arm], alpha=.8, label=f'Seed {seed}')
    ax.axvline(3.5, color='#cccccc', lw=.8); ax.set_ylim(35, 105); ax.set_title(title, fontsize=9)
    ax.set_xticks(range(6), ['nm', 'µm', 'sci.', 'eV', 'nm plot', 'eV plot'], rotation=40, ha='right'); ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=7, ncol=2, loc='lower left')
axes[0].set_ylabel('Compare accuracy (%)')
fig.savefig(OUT / 'figures' / 'lora_transfer.pdf'); plt.close(fig)

# Figure 4: paired scope measures for every seed, with instance-bootstrap intervals and unadapted references.
fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), sharex=True, sharey=True, layout='constrained')
for ax, name, title in zip(axes, SETS[-2:], ['MOCLIP tables', 'Gold tables']):
    ax.axhline(0, lw=.8, c='#AAAAAA'); ax.axvline(0, lw=.8, c='#AAAAAA')
    for arm in ALL_ARMS:
        for seed, model in enumerate(arm_models(arm)):
            v = get_suff(name, model, 'table')
            xd, yd = 100 * v.discrimination, 100 * v.outside_delta
            ax.errorbar(xd, yd, xerr=[[xd - 100 * v.d_lo], [100 * v.d_hi - xd]], yerr=[[yd - 100 * v.o_lo], [100 * v.o_hi - yd]],
                        fmt='none', ecolor=COLORS[arm], elinewidth=.6, alpha=.45, zorder=2)
            ax.scatter(xd, yd, c=COLORS[arm], marker=['o', 's', '^'][seed], s=40, label=arm if seed == 0 else None, zorder=3,
                       edgecolors='black' if arm in NEW_ARMS else 'none', linewidths=.6)
    for model, mk in [('d3-nano', 'P'), ('luna-chat-low', 'X')]:
        v = get_suff(name, model, 'table')
        ax.scatter(100 * v.discrimination, 100 * v.outside_delta, marker=mk, s=60, facecolors='none', edgecolors='#333333',
                   label=('Base d3-nano' if model == 'd3-nano' else 'Luna gen. low'), zorder=4)
    ax.scatter(100, 0, marker='*', s=100, c='black', label='Ideal', zorder=2)
    ax.set_title(title); ax.set_xlabel('Scope discrimination (percentage points)'); ax.set_xlim(-20, 105); ax.set_ylim(-106, 30); ax.grid(alpha=.15)
axes[0].set_ylabel('Outside-cut accuracy − full accuracy (pp)'); axes[1].legend(ncol=4, fontsize=7, loc='lower center')
fig.savefig(OUT / 'figures' / 'scope_training.pdf'); plt.close(fig)

print('wrote', OUT)

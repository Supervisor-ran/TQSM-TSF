"""TQSM balanced selector should preserve the 5+5 paper budget."""
from src.data_collector.topk_generator import TopKGenerator

def test_balanced_five_each():
    generator=TopKGenerator(debug=False)
    records=[dict(scope=s,month='2026-01',abstract=f'{s}{i}',score_full=1.0-i/100)
             for s in ('macro','micro') for i in range(7)]
    # Use exact sorting key shape from current source; only size/scope is tested here.
    selected=generator.select_records(records,k=10,variant='full',mode='balanced')
    assert len(selected)==10
    assert sum(x['scope']=='macro' for x in selected)==5
    assert sum(x['scope']=='micro' for x in selected)==5

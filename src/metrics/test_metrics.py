import copy
from src.metrics.tracker_metrics import EvaluatorMOT

def create_dummy_trajectory(num_frames=10, obj_id=1, x_start=10):
    traj = []
    for f in range(1, num_frames + 1):
        traj.append({
            'frame': f,
            'id': obj_id,
            'bb_left': x_start + f * 2,
            'bb_top': 20,
            'bb_width': 10,
            'bb_height': 10
        })
    return traj

def test_metrics():
    evaluator = EvaluatorMOT(iou_threshold=0.5)

    # Caso (a): Predição Perfeita
    gt_a = create_dummy_trajectory(num_frames=10, obj_id=1)
    pred_a = copy.deepcopy(gt_a)
    
    res_a = evaluator.evaluate(gt_a, pred_a)
    print(f"[Caso A] Identico - IDF1: {res_a['IDF1']:.2f} (Esperado: 1.0), IDSW: {res_a['IDSW']} (Esperado: 0)")
    assert res_a['IDF1'] == 1.0 and res_a['IDSW'] == 0

    # Caso (b): Troca de ID a partir do quadro 5
    gt_b = create_dummy_trajectory(num_frames=10, obj_id=1)
    pred_b = []
    for item in gt_b:
        new_item = copy.deepcopy(item)
        if item['frame'] >= 5:
            new_item['id'] = 99  # ID trocado
        pred_b.append(new_item)
        
    res_b = evaluator.evaluate(gt_b, pred_b)
    print(f"[Caso B] ID Switch - IDF1: {res_b['IDF1']:.2f}, IDSW: {res_b['IDSW']} (Esperado: 1)")
    assert res_b['IDSW'] == 1

    # Caso (c): Track Partida no meio (Gap entre os quadros 4 e 7)
    gt_c = create_dummy_trajectory(num_frames=10, obj_id=1)
    # O rastreador perde a track nos quadros 4, 5, 6
    pred_c = [copy.deepcopy(item) for item in gt_c if item['frame'] not in [4, 5, 6]]

    res_c = evaluator.evaluate(gt_c, pred_c)
    print(f"[Caso C] Track Partida - IDF1: {res_c['IDF1']:.2f}, Frag: {res_c['Frag']} (Esperado: 1)")
    assert res_c['Frag'] == 1

    print("\n✅ Todos os testes sintéticos das métricas passaram com sucesso!")

if __name__ == "__main__":
    test_metrics()
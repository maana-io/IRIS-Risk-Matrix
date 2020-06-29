
import numpy as np
from scipy.stats import entropy
from app.logic.helpers import is_low_confidence, calculate_score, iris_unique_id, id


def flattenRiskToDataset(risks):

    features = []
    field_datas = []
    for _ in range(12):
        field_datas.append([])

    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 0,
            'name': 'id',
            'type': "TEXT"
        },
        'data': field_datas[0]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 1,
            'name': 'title',
            'type': "TEXT"
        },
        'data': field_datas[1]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 2,
            'name': 'description',
            'type': "TEXT"
        },
        'data': field_datas[2]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 3,
            'name': 'cause',
            'type': "TEXT"
        },
        'data': field_datas[3]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 4,
            'name': 'consequence',
            'type': "TEXT"
        },
        'data': field_datas[4]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 5,
            'name': 'topology.id',
            'type': "CATEGORICAL"
        },
        'data': field_datas[5]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 6,
            'name': 'topology.onshoreOffshore',
            'type': "CATEGORICAL"
        },
        'data': field_datas[6]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 7,
            'name': 'topology.upstreamDownstream',
            'type': "CATEGORICAL"
        },
        'data': field_datas[7]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 8,
            'name': 'topology.oilGas',
            'type': "CATEGORICAL"
        },
        'data': field_datas[8]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 9,
            'name': 'topology.facilityType',
            'type': "CATEGORICAL"
        },
        'data': field_datas[9]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 10,
            'name': 'discipline.id',
            'type': "CATEGORICAL"
        },
        'data': field_datas[10]
    })
    features.append({
        'id': id(),
        'feature': {
            'id': id(),
            'index': 11,
            'name': 'discipline.name',
            'type': "CATEGORICAL"
        },
        'data': field_datas[11]
    })


    for risk in risks:
        field_datas[0].append({'id': id(), 'text': risk['id']})
        field_datas[1].append({'id': id(), 'text': risk['title']})
        field_datas[2].append({'id': id(), 'text': risk['description']})
        field_datas[3].append({'id': id(), 'text': risk['cause']})
        field_datas[4].append({'id': id(), 'text': risk['consequence']})

        field_datas[5].append({'id': id(), 'text': risk['topology']['id']})
        field_datas[6].append({'id': id(), 'text': risk['topology']['onshoreOffshore']})
        field_datas[7].append({'id': id(), 'text': risk['topology']['upstreamDownstream']})
        field_datas[8].append({'id': id(), 'text': risk['topology']['oilGas']})
        field_datas[9].append({'id': id(), 'text': risk['topology']['facilityType']})

        field_datas[10].append({'id': id(), 'text': risk['discipline']['id']})
        field_datas[11].append({'id': id(), 'text': risk['discipline']['name']})

    return {
        'id': iris_unique_id(),
        'features': features
    }



def flattenLabeledRiskToDataset(labeled_risks):
    allRisks = [ labeled_risk['risk'] for labeled_risk in labeled_risks]
    labels = [ { 'id': id(), 'text': ' '.join([labeled_risk['severity'], labeled_risk['likelihood']]) }
               for labeled_risk in labeled_risks]

    label_feature = {
        'id': id(),
        'feature': {
            'id': id(),
            'index': 12,
            'name': 'severity likelihood',
            'type': "LABEL"
        },
        'data': labels
    }
    dataset = flattenRiskToDataset(allRisks)

    return {
        'id': iris_unique_id(),
        'data': dataset,
        'label': label_feature
    }



def classificationResultToClassifiedRisk(classification_result):
    input_data = classification_result['dataInstance']['dataset']['features']
    risk = {}
    topology = {}
    discipline = {}
    for feat_data in input_data:
        if feat_data['feature']['name'] in ['id', 'title', 'description', 'cause', 'consequence']:
            risk.update({feat_data['feature']['name']: feat_data['data'][0]['text']})
        elif feat_data['feature']['name'].startswith('topology.'):
            topology.update({feat_data['feature']['name'].replace('topology.', ''): feat_data['data'][0]['text']})
        elif feat_data['feature']['name'].startswith('discipline.'):
            discipline.update({feat_data['feature']['name'].replace('discipline.', ''): feat_data['data'][0]['text']})
    risk.update({'topology': topology})
    risk.update({'discipline': discipline})

    severity, likelihood = classification_result['predictedLabel']['label'].split()
    entropy = classification_result['entropy']
    classified_risk = {
        'id': id(),
        'risk': risk,
        'severity':severity,
        'likelihood': likelihood,
        'confidenceLevel': entropy,
        'lowConfidence': is_low_confidence(entropy),
        'score': calculate_score(severity, likelihood),
        'contributors': classification_result['contributors'],
        'recommends': classification_result['recommends']
    }

    return classified_risk





def batchClassificationResultToRiskProfile(batch_classification_result, profile_id):
    max_entr = -1
    for cls_sum in batch_classification_result['classSummaries']:
        max_entr = max(max_entr, max(cls_sum['entropies']))

    risk_scores = []
    risk_buckets = []
    for class_summary in batch_classification_result['classSummaries']:
        severity, likelihood = class_summary['label'].split()
        risks = []
        for res in class_summary['results']:
            res['entropy'] /= max_entr
            classifiedRisk = classificationResultToClassifiedRisk(res)
            risks.append(classifiedRisk)
            risk_scores.append(classifiedRisk['score'])
        bucket = {
            'id': id(),
            'severity': severity,
            'likelihood': likelihood,
            'numberOfRisks': class_summary['numInstances'],
            'averageConfidenceLevel': np.average(class_summary['entropies'])/max_entr,
            'numberOfLowConfidenceRisks': len([entropy for entropy in class_summary['entropies'] if is_low_confidence(entropy/max_entr)]),
            'risks': risks
        }
        risk_buckets.append(bucket)

    return {
        'id': profile_id,
        'compoundRisk': np.average(risk_scores),
        'riskBuckets': risk_buckets
    }


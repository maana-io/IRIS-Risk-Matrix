
import uuid
import numpy as np

from app.settings import LOW_CONFIDENCE_THRESHOLD, IRIS_UNIQUE_ID, IRIS_SEVERITY_LIKELIHOOD_MODEL_ID



def id():
    return str(uuid.uuid4())


def iris_unique_id():
    return IRIS_UNIQUE_ID


def iris_severity_likelihood_model_id():
    return IRIS_SEVERITY_LIKELIHOOD_MODEL_ID



def calculate_score(severity, likelihood):
    return 1.0 * int(severity) * int(likelihood)/9.0



def is_low_confidence(confidence_level):
    return confidence_level < LOW_CONFIDENCE_THRESHOLD



def mergeBucket(merge_this, to_this):
    assert merge_this is not None
    assert to_this is not None
    assert merge_this['severity'] == to_this['severity']
    assert merge_this['likelihood'] == to_this['likelihood']

    current_risk_map = {}
    for ridx, clf_risk in enumerate(to_this['risks']):
        current_risk_map.update({clf_risk['risk']['id']: ridx})

    for clf_risk in merge_this['risks']:
        risk_id = clf_risk['risk']['id']
        if risk_id not in current_risk_map:
            to_this['risks'].append(clf_risk)
            current_risk_map.update({risk_id: len(to_this['risks'])-1})
        else:
            ridx = current_risk_map[risk_id]
            to_this['risks'][ridx] = clf_risk

    to_this['numberOfRisks'] = len(to_this['risks'])
    to_this['numberOfLowConfidenceRisks'] = len([risk for risk in to_this['risks'] if risk['lowConfidence']])
    to_this['averageConfidenceLevel'] = np.average([risk['confidenceLevel'] for risk in to_this['risks']])

    return to_this



def mergeRiskProfiles(current_profile, new_profile):
    if current_profile is None:
        return new_profile
    if new_profile is None:
        return current_profile

    current_buckets_map = {}
    for bidx, bucket in enumerate(current_profile['riskBuckets']):
        current_buckets_map.update({bucket['severity'] + bucket['likelihood']: bidx})

    for new_bucket in new_profile['riskBuckets']:
        bucket_key = new_bucket['severity'] + new_bucket['likelihood']
        if bucket_key in current_buckets_map:
            current_bucket_idx = current_buckets_map[bucket_key]
            current_bucket = current_profile['riskBuckets'][current_bucket_idx]
            mergeBucket(new_bucket, current_bucket)
        else:
            current_profile['riskBuckets'].append(new_bucket)

    #recalculate compound factor
    risk_scores = [ risk['score'] for bucket in current_profile['riskBuckets'] for risk in bucket['risks']]
    current_profile['compoundRisk'] = np.average(risk_scores)
    return current_profile

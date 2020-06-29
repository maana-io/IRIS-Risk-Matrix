
import logging
from app.logic.adaptors import *
from app.logic.helpers import *

logger = logging.getLogger(__name__)

#
# graphql client = info.context["client"]
#
resolvers = {
    'Query': {
        'flattenRisks': lambda value, info, **args: flattenRiskToDataset(args['risks']),
        'flattenLabeledRisks': lambda value, info, **args: flattenLabeledRiskToDataset(args['risks']),
        'classificationResultsToRiskProfile': lambda value, info, **args:
                    batchClassificationResultToRiskProfile(args['results'], args['profile_id']),
        'irisUniqueID':  lambda value, info, **args: iris_unique_id(),
        'irisSeverityLikelihoodModelID':  lambda value, info, **args: iris_severity_likelihood_model_id(),
        'mergeRiskProfiles': lambda value, info, **args: mergeRiskProfiles(args.get('current', None),  args.get('new', None)),
    },
    'Mutation': {
    },
    'Object': {
    },
    'Scalar': {
    },
}






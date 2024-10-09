import torch
import sys
sys.path.insert(0, '..')
from mbank.flow import GW_Flow, TanhTransform, STD_GW_Flow
import re

def infer_STD_GW_Flow_params(w):
	"""
	Given the weigth dicitionary (loaded with torch.load), it infers the dimensionality ```D```, the number of layers ```n_layers``` and the number of hidden features ```hidden_features``` of the flow
	
	Parameters
	----------
		w: dict
			Dictionary with torch weigths (loaded with torch.load)
	
	Returns
	-------
		D: int
			Dimensionality of the flow
	
		n_layers: int
			Number of layers
			
		hidden_features: int
			Number of hidden features of the flow
	"""
	try:
		D = w['_transform._transforms.0.low'].shape[0]
		n_layers = re.findall(r'\d+', list(w.keys())[-1])
		assert len(n_layers)==1
		n_layers = int(n_layers[0])//2

		hidden_features = '_transform._transforms.{}.autoregressive_net.final_layer.mask'.format(2*n_layers)
		hidden_features = w[hidden_features].shape[1]
	except:
		raise ValueError("The given weight dictionary does not match the architecture of a `STD_GW_Flow`")

	return D, n_layers, hidden_features



filename = 'test_load_weigths/weights'

D_true, n_layers_true, hidden_features_true = 40, 20, 4
flow = STD_GW_Flow(D_true, n_layers_true, hidden_features_true)

flow.save(filename)

w = torch.load(filename)
#print(len(w))
#print(w.keys())

D, n_layers, hidden_features = infer_STD_GW_Flow_params(w)

print(D_true, n_layers_true, hidden_features_true)
print(D, n_layers, hidden_features)

flow_new = STD_GW_Flow.load_flow(filename)
flow_new.load(filename)






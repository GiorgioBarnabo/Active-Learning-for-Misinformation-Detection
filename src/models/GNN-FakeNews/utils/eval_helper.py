from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score

"""
	Utility functions for evaluating the model performance
"""
import torch
import torch.nn.functional as F
from sklearn import metrics


def eval_deep(log, loader):
	"""
	Evaluating the classification performance given mini-batch data
	"""

	# get the empirical batch_size for each mini-batch
	data_size = len(loader.dataset.indices)
	batch_size = loader.batch_size
	if data_size % batch_size == 0:
		size_list = [batch_size] * (data_size//batch_size)
	else:
		size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]

	assert len(log) == len(size_list)

	accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch, size in zip(log, size_list):
		pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
		prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		label_log.extend(y)

		accuracy += accuracy_score(y, pred_y) * size
		f1_macro += f1_score(y, pred_y, average='macro') * size
		f1_micro += f1_score(y, pred_y, average='micro') * size
		precision += precision_score(y, pred_y, zero_division=0) * size
		recall += recall_score(y, pred_y, zero_division=0) * size

	auc = roc_auc_score(label_log, prob_log)
	ap = average_precision_score(label_log, prob_log)

	return accuracy/data_size, f1_macro/data_size, f1_micro/data_size, precision/data_size, recall/data_size, auc, ap

def compute_metrics(pred_test, y_test):
    (pre_0,pre_1),(rec_0,rec_1),(f_0,f_1),(_,_) = metrics.precision_recall_fscore_support(y_test,pred_test, labels=[0,1],zero_division=0)
    acc = metrics.accuracy_score(y_test,pred_test)

    res = [acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0] #, auc]

    return res


@torch.no_grad()
def new_compute_test(model, loader, args, verbose=False):
	model.eval()
	loss_test = 0.0
	pred_y = []
	real_y = []
	for data in loader:
		if not args.multi_gpu:
			data = data.to(args.device)
		out = model(data)
		if args.multi_gpu:
			y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
		else:
			y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		pred_y += list(F.softmax(out, dim=1).cpu().numpy()[:,1]>0.5)
		real_y += list(y.cpu().numpy())

	return compute_metrics(pred_y, real_y)
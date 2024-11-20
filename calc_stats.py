import sys
import numpy as np
import pandas as pd

file1 = sys.argv[1] #Ingroup output
file2 = sys.argv[2] #OOD output
code = sys.argv[3] if len(sys.argv) > 3 else "out" #Identification code

X = pd.read_table(file1)
O = pd.read_table(file2)

X["uncertainty"] = -X["uncertainty"]
O["uncertainty"] = -O["uncertainty"]

X["energy"] = -X["energy"]
O["energy"] = -O["energy"]

#Calculate baseline accuracy
acc = []
for l in [650, 300, 150]:
	for k in range(20):
		x = X[np.logical_and(X["length"]==l, X["k"]==k)]
		
		Acc0 = np.sum(x["true_class"]==x["pred_class1"])/x.shape[0]
		
		acc.append([l, k, Acc0])
acc = pd.DataFrame(acc, columns=("l", "k", "Acc"))
#print (acc)

#Calculate FNR
fnr = []
for l in [650, 300, 150]:
	for k in range(20):
		x = X[np.logical_and(X["length"]==l, X["k"]==k)]
		o = O[np.logical_and(O["length"]==l, O["k"]==k)]
		
		#Class-wise quantiles for three metrics
		q_p = [[i, np.quantile(x["prob"], q=0.05)] for i, x in x.groupby(by="true_class")]
		q_e = [[i, np.quantile(x["energy"], q=0.05)] for i, x in x.groupby(by="true_class")]
		q_u = [[i, np.quantile(x["uncertainty"], q=0.05)] for i, x in x.groupby(by="true_class")]
		
		q_p = np.array(q_p)
		q_e = np.array(q_e)
		q_u = np.array(q_u)
		
		#Compare each entry with 95% threshold
		o_q_p = np.array([q_p[q_p[:,0]==i,1][0] for i in o["pred_class1"]])
		o_q_e = np.array([q_e[q_e[:,0]==i,1][0] for i in o["pred_class1"]])
		o_q_u =  np.array([q_u[q_u[:,0]==i,1][0] for i in o["pred_class2"]])
		
		err_p = np.sum(o["prob"] >= o_q_p)/o.shape[0]
		err_e = np.sum(o["energy"] >= o_q_e)/o.shape[0]
		err_u = np.sum(o["uncertainty"] >= o_q_u)/o.shape[0]
		
		#Majority voting
		tab = np.vstack((o["prob"] >= o_q_p, o["energy"] >= o_q_e, o["uncertainty"] >= o_q_u))
		vote = np.sum(tab.T, axis=1)
		err_mv = np.sum(vote >= 2)/o.shape[0]
		
		fnr.append([l, k, err_p, err_e, err_u, err_mv])
		
fnr = pd.DataFrame(fnr, columns=("l", "k", "Err.p", "Err.e", "Err.u", "Err.mv"))

res = pd.merge(acc, fnr)
res.insert(0, "code", [code for i in range(res.shape[0])], True)

#print(res)
res.to_csv(f"stats.{code}.txt", sep="\t", index=False)


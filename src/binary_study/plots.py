"""Standalone external-study figures from locked predictions, without model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score,confusion_matrix

LABELS={'linear':'B0: logistic','concat':'B1: fixed fusion','concat_subtype':'B2: subtype supervision',
        'reliability':'B3: reliability fusion','reliability_subtype':'B4: combined candidate'}


def draw(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    root=Path(root);result=json.loads((root/'external_results.json').read_text())
    frame=pd.read_csv(root/'external_predictions.csv')
    selected=frame[(frame.track=='PAD/external/clinical_metadata')&(frame.condition=='clean')]
    if selected.empty:raise ValueError('No clean locked external clinical-plus-metadata predictions')
    out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for variant,g in selected.groupby('variant',sort=False):
        if len(g.target.unique())<2:continue
        fpr,tpr,_=roc_curve(g.target,g.probability);precision,recall,_=precision_recall_curve(g.target,g.probability)
        axes[0].plot(fpr,tpr,label=f'{LABELS[variant]} ({roc_auc_score(g.target,g.probability):.3f})')
        axes[1].plot(recall,precision,label=f'{LABELS[variant]} ({average_precision_score(g.target,g.probability):.3f})')
    axes[0].plot([0,1],[0,1],':',color='grey');axes[1].axhline(selected.groupby('case_id').target.first().mean(),ls=':',color='grey')
    axes[0].set(xlabel='False-positive rate',ylabel='Sensitivity',title='External ROC (legend: AUC)')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='External precision-recall (legend: AP)')
    for ax in axes:ax.legend(fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    fig.savefig(out/'external_roc_pr.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,variant in zip(axes,['concat','reliability_subtype']):
        g=selected[selected.variant==variant];cm=confusion_matrix(g.target,g.predicted,labels=[0,1]);ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#163448',fontsize=16)
        ax.set_xticks([0,1],['Nonmalignant','Malignant']);ax.set_yticks([0,1],['Nonmalignant','Malignant'])
        ax.set(xlabel='Predicted',ylabel='Reference',title=LABELS[variant])
    fig.suptitle('PAD external test: independently locked thresholds')
    fig.savefig(out/'external_confusions.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for variant in ['concat','reliability_subtype']:
        g=selected[selected.variant==variant];fraction,mean=calibration_curve(g.target,g.probability,n_bins=10,strategy='uniform')
        ax.plot(mean,fraction,'o-',label=LABELS[variant])
    ax.plot([0,1],[0,1],':',color='grey');ax.set(xlabel='Predicted probability',ylabel='Observed malignant fraction',title='External calibration; no refitting on PAD')
    ax.legend();fig.savefig(out/'external_calibration.png',dpi=180);plt.close(fig)
    print(f'External evaluation figures saved in {out}',flush=True)
    return sorted(out.glob('external_*.png'))

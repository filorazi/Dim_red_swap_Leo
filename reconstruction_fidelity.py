from utils import * 
import pennylane as qml
import os 
from autoencoder8 import *

def main():
    fid_fold = 'out_fi'
    emd_fold = 'out_em'

    binder_fi = {}

    for subfolder in os.listdir(fid_fold):
        mq = int(subfolder[4])-int(subfolder[-1])
        subfolder_path = os.path.join(fid_fold, subfolder)
        binder_fi[mq]=[]
        for batchfolder in os.listdir(subfolder_path):
            batchfolder_path = os.path.join(subfolder_path, batchfolder)
            for file in os.listdir(batchfolder_path):
                file_path= os.path.join(batchfolder_path, file)
                if 'weights' in file:
                    weight_file = os.path.join(batchfolder_path, file)
                    append = file[7:-3]
                    tloss_file = os.path.join(batchfolder_path, f'loss_train_{append}npy')
                    vloss_file = os.path.join(batchfolder_path, f'loss_val{append}npy')
                    binder_fi[mq].append((weight_file, tloss_file,vloss_file,file))


    binder_em = {}

    for subfolder in os.listdir(emd_fold):
        mq = int(subfolder[4])-int(subfolder[-1])
        subfolder_path = os.path.join(emd_fold, subfolder)
        binder_em[mq]=[]
        for batchfolder in os.listdir(subfolder_path):
            batchfolder_path = os.path.join(subfolder_path, batchfolder)
            for file in os.listdir(batchfolder_path):
                file_path= os.path.join(batchfolder_path, file)
                if 'weights' in file:
                    weight_file = os.path.join(batchfolder_path, file)
                    append = file[7:-3]
                    tloss_file = os.path.join(batchfolder_path, f'loss_train_{append}npy')
                    vloss_file = os.path.join(batchfolder_path, f'loss_val{append}npy')
                    binder_em[mq].append((weight_file, tloss_file,vloss_file,file))



    data = get_data(8)
    X=[data['ground_states'][x] for x in range(100)]


    res = pd.DataFrame(columns=['vloss','tloss','fidelity','file','train_type','mq','data_idx'])
    for mq in range(4,8):
        for model in binder_em[mq]:
            n_qubit = 8
            t_qubit = n_qubit-mq
            dvc = qml.device('default.mixed', wires=n_qubit+t_qubit, shots=None)
            @qml.qnode(dvc)
            def circ(ae,dm1):
                qml.StatePrep(dm1, wires=range(0,n_qubit))
                wire_map=dict(zip(list(range(t_qubit)),list(range(n_qubit,n_qubit+t_qubit))))
                ae.get_cirq(0)
                qml.adjoint(qml.map_wires(ae.get_cirq, wire_map))(0)
                return qml.density_matrix(list(range(t_qubit, len(dvc.wires)+1)))

            dvc = qml.device('default.mixed', wires=n_qubit, shots=None)
            ae = Axutoencoder(n_qubit,t_qubit,dvc,'c11')
            ae.set_layers(3)
            ae.set_weights_loss(np.load(model[0]), np.load(model[1]))
            # qml.draw_mpl(circ)(ae,X[0])
            # fidel = []
            for i in range(100):
                # fidel.append(qml.math.fidelity( circ(ae,X[i]),np.outer(X[i], np.conj(X[i]))))
                fidel= qml.math.fidelity( circ(ae,X[i]),np.outer(X[i], np.conj(X[i])))
                timeindex = np.argmin(np.load(model[2]))+1
                vloss = np.min(np.load(model[2]))
                tloss = np.min(np.load(model[1]))
                d = {'vloss':vloss,
                    'tloss':tloss,
                    'fidelity':fidel,
                    'file':model[3],
                    'train_type':'em',
                    'mq':mq,
                    'data_idx':i}
                res= pd.concat([res,pd.DataFrame([d])])

    res.to_csv('./reconstruction_results.csv')

if __name__ == '__main__':
    main()
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


    for mq in range(7,0,-1):
        for model in binder_fi[mq]:
            res = pd.DataFrame(columns=['vloss','tloss','fidelity','file','train_type','mq','data_idx'])

            n_qubit = 8
            t_qubit = n_qubit-mq
            def encoder_decoder(ae,dm1):
                dvc2 = qml.device('default.mixed', wires=n_qubit, shots=None)
                
                @qml.qnode(dvc2)
                def encoder(ae,state):
                    qml.StatePrep(state, wires=range(0,n_qubit))
                    ae.get_cirq(0)
                    return qml.density_matrix(list(range(t_qubit, len(dvc2.wires))))

                dvc3 = qml.device('default.mixed', wires=n_qubit, shots=None)

                @qml.qnode(dvc3)
                def decoder(ae,dm):
                    qml.QubitDensityMatrix(dm, wires=range(t_qubit,len(dvc2.wires)))
                    qml.adjoint(ae.get_cirq)(0)
                    return qml.state()

                return decoder(ae,encoder(ae,dm1))

        
            dvc1 = qml.device('default.mixed', wires=n_qubit, shots=None)
            ae = Axutoencoder(n_qubit,t_qubit,dvc1,'c11')
            ae.set_layers(3)
            ae.set_weights_loss(np.load(model[0]), np.load(model[1]))
            # qml.draw_mpl(circ)(ae,X[0])
            # fidel = []
            list_op_support=[1,2,3]
            list_op_support_probs=[1., 1., 1.]
            list_op_support_max_range=[1, 5, 3]
            i =1
            set_global( n_qubit,
                n_qubit,
                t_qubit,
                list_op_support[:mq],
                list_op_support_probs[:mq],
                False,
                list_op_support_max_range[:mq],
                use_jax=False)

            for i in range(0,100,4):
                # fidel.append(qml.math.fidelity( circ(ae,X[i]),np.outer(X[i], np.conj(X[i]))))
                EM_dist= cost__EM( [np.outer(X[i], np.conj(X[i]))])([encoder_decoder(ae,X[i])])
                print(EM_dist)
                timeindex = np.argmin(np.load(model[2]))+1
                vloss = np.min(np.load(model[2]))
                tloss = np.min(np.load(model[1]))
                d = {'vloss':vloss,
                    'tloss':tloss,
                    'EM_dist':EM_dist,
                    'file':model[3],
                    'train_type':'fi',
                    'mq':mq,
                    'data_idx':i}
                res= pd.concat([res,pd.DataFrame([d])])
            print(f'model {model} completed')
            res.to_csv(f'./reconstruction_em_results_{mq}_{model[3][7:-4]}.csv')

if __name__ == '__main__':
    main()
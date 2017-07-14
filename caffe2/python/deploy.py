import os
import pickle

from caffe2.python import workspace, core

def DeployNets(init_net, pred_net, ins, outs):
    
    """
    Removes training operators and adds parameter initialization based on
    the current workspace. 
    
    Args:
        init_net:   net that initializes parameters
        
        pred_net:   net used for prediction
        
        ins:        list of BlobReferences or str's that are to be the inputs
                    of the new pred_net
                    
        outs:       list of BlobReferences or str's that are to be the outputs
                    of the new pred_net
                    
    example usage: init_net, pred_net = DeployNets(model.param_init_net, model.net, ['data'], ['softmax'])
    """
    
    # preserve names
    init_name = init_net.Proto().name
    pred_name = pred_net.Proto().name
    
    pred_net, _ = pred_net.ClonePartial('', ins, outs)
    #pred_net = core_ext.Net(pred_net.Proto())
    
    #init_net = core_ext.Net(init_name)
    for blob in list(pred_net.ExternalInputBlobs() - set(ins)):
        
        val = workspace.FetchBlob(blob)
        init_net.Const(val, blob)
    
    init_net.Proto().name = init_name
    pred_net.Proto().name = pred_name
        
    return init_net, pred_net 

def SaveNets(nets, folder=''):
    
    if not os.path.exists(folder):
            os.makedirs(folder)
        
    names = []
    
    for net in nets:
        name = net.Proto().name
        filename = name+'.pb'
        path = os.path.join(folder, filename)
        print('Saving {} as {}'.format(net.Proto().name, path))
        f = open(path, 'w')
        pickle.dump(net.Proto(), f)
        names += [name]
    
    print('Save Completed.')
    
    return folder, names
              

def LoadNets(folder, names):
    
    assert os.path.exists(folder)
    
    nets = ()
    
    for name in names:
        
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            path += '.pb'
            
        if os.path.exists(path):
            print('Loading {}'.format(path))
        
        f = open(path, 'r')
        pb = pickle.load(f)
        net = core.Net(pb)
        net.Proto().name = name.split('.')[0]
        
        nets += (net,)
    
    return nets

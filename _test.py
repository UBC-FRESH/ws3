
def import_woodstockmodel(load_pickle=False, dump_pickle=False):
    from ws3.woodstock import WoodstockModel
    import cPickle
    import time
        
    model_name = 'PC_4537_U03153_4FF_V05'
    model_path = '../../dat/woodstock/03153'
    
    if load_pickle:
        print 'loading pickle file...'
        t = time.time()
        wm = cPickle.load(open('wm.p', 'r'))
        print("--- %i seconds ---" % (time.time() - t))
    else:
        wm = WoodstockModel(model_name, model_path)
        print 'importing CONSTANTS section'
        wm.import_constants_section()
        print 'importing LANDSCAPE section'
        wm.import_landscape_section()
        print 'importing AREAS section'
        wm.import_areas_section()
        print 'importing YIELDS section'
        wm.import_yields_section()
        print 'importing ACTIONS section'
        wm.import_actions_section()
        print 'importing TRANSITIONS section'
        wm.import_transitions_section()
        print 'importing OUTPUTS section'
        wm.import_outputs_section()
        if dump_pickle:
            print 'dumping pickle file...'
            t = time.time()
            cPickle.dump(wm, open('wm.p', 'wb'), 2) # use
            print("--- %i seconds ---" % (time.time() - t))
    return wm
    
    
if __name__ == '__main__':
    import os
    import sys
    #import time
    
    try:
        _path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    except:
        _path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
    if not _path in os.sys.path:
        os.sys.path.insert(1, _path)
    del _path

    load_pickle = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dump_pickle = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    #start_time = time.time()
    wm = import_woodstockmodel(load_pickle, dump_pickle)
    #print("--- %i seconds ---" % (time.time() - start_time))
    

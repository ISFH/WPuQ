class H5ls:
    '''
    A class to extract all nodes from an H5py file.

    Returns
    -------

    '''

    def __init__(self):
        # Store an empty list for dataset names
        self.names = []

    def __call__(self, name, node):
        '''
        Call

        Parameters
        ----------
        name : str
            The name of the node
        node : h5py.Dataset or h5py.Group
            The node

        Returns
        -------

        '''
        # only h5py datasets have dtype attribute, so we can search on this
        if hasattr(node, 'dtype') and name not in self.names:
            self.names += [name]

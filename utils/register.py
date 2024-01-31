r"""A kernel module that contains a global register for unified model, dataset, and pre-training algorithms access.
"""

class Register(object):
    r"""
    Global register for unified model, dataset, and pre-training algorithms access.
    """

    def __init__(self):
        self.pipelines = dict()
        self.launchers = dict()
        self.models = dict()
        self.datasets = dict()
        self.dataloader = dict()
        self.ood_algs = dict()
        self.encoders = dict()

    def pipeline_register(self, pipeline_class):
        r"""
        Register for pipeline access.

        Args:
            pipeline_class (class): pipeline class

        Returns (class):
            pipeline class

        """
        self.pipelines[pipeline_class.__name__] = pipeline_class
        return pipeline_class

    def launcher_register(self, launcher_class):
        r"""
        Register for pipeline access.

        Args:
            launcher_class (class): pipeline class

        Returns (class):
            pipeline class

        """
        self.launchers[launcher_class.__name__] = launcher_class
        return launcher_class

    def model_register(self, model_class):
        r"""
        Register for model access.

        Args:
            model_class (class): model class

        Returns (class):
            model class

        """
        self.models[model_class.__name__] = model_class
        return model_class
    
    def encoder_register(self, encoder_class):
        r"""
        Register for model access.

        Args:
            model_class (class): model class

        Returns (class):
            model class

        """
        self.encoders[encoder_class.__name__] = encoder_class
        return encoder_class

    def dataset_register(self, dataset_class):
        r"""
        Register for dataset access.

        Args:
            dataset_class (class): dataset class

        Returns (class):
            dataset class

        """
        self.datasets[dataset_class.__name__] = dataset_class
        return dataset_class

    def dataloader_register(self, dataloader_class):
        r"""
        Register for dataloader access.

        Args:
            dataloader_class (class): dataloader class

        Returns (class):
            dataloader class

        """
        self.dataloader[dataloader_class.__name__] = dataloader_class
        return dataloader_class




register = Register()  #: The register object used for accessing models, datasets and pre-training algorithms.

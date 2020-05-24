import torch
import torch.nn as nn
from torch.autograd import Function

from torch.hub import load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class AlexNetDANN(nn.Module):

	def __init__(self, num_classes=1000):
		super(AlexNetDANN, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

		self.domain_classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 2),
		)
		

	# DEFINE HOW FORWARD PASS IS COMPUTED
	def forward(self, x, alpha=None, dest='classifier'):
		"""
		# PYTORCH ALEXNET IMPLEMENTATION
			x = self.features(x)
			x = self.avgpool(x)
			x = torch.flatten(x, 1)
			x = self.classifier(x)
		"""

		"""
		# HW3 SUGGESTION OF IMPLEMENTATION
			features = self.features
	        features = features.view(features.size(0), -1)
	        if alpha is not None:
		        reverse_feature = ReverseLayerF.apply(features, alpha)
		        discriminator_output = ...
		        return discriminator_output
	        else:
	            class_outputs = ...
	            return class_outputs
		"""

		"""
		# PYTORCH DANN IMPLEMENTATION		
			input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
	        feature = self.feature(input_data)
	        feature = feature.view(-1, 50 * 4 * 4)
	        reverse_feature = ReverseLayerF.apply(feature, alpha)
	        class_output = self.class_classifier(feature)
	        domain_output = self.domain_classifier(reverse_feature)
        """

		x = self.features(x)
		x = self.avgpool(x)
		features = torch.flatten(x, 1)

		if dest == 'classifier':
			output = self.classifier(features)
			return output

		elif dest == 'domain_classifier':
			if alpha == None:
				print('FATAL ERROR - Attach a valid alpha when forwarding to the domain classifier')
				sys.exit()

			reverse_features = ReverseLayerF.apply(features, alpha)
			domain_output = self.domain_classifier(reverse_features)
			return domain_output

		else:
			print('FATAL ERROR - Invalid parameters to forward function in AlexNetDANN')
			sys.exit()

def alexnetDANN(pretrained=True, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetDANN(num_classes=1000, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
	
	# Change output classes
    model.classifier[6] = nn.Linear(4096, 7)
    model.domain_classifier[6] = nn.Linear(4096, 2)

    # Copy pretrained weights from the classifier to the domain_classifier
    model.domain_classifier[1].weight.data = model.classifier[1].weight.data.clone()
    model.domain_classifier[1].bias.data = model.classifier[1].bias.data.clone()

    model.domain_classifier[4].weight.data = model.classifier[4].weight.data.clone()
    model.domain_classifier[4].bias.data = model.classifier[4].bias.data.clone()

    return model
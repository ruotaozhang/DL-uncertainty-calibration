import torch
import torchvision.models as models


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GlobalCatAvgMaxPool(torch.nn.Module):
    def __init__(self, kernel_size):
        super(GlobalCatAvgMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.globalAvgPool = torch.nn.AvgPool2d(kernel_size=self.kernel_size)
        self.globalMaxPool = torch.nn.MaxPool2d(kernel_size=self.kernel_size)

    def forward(self, x):
        out1 = self.globalAvgPool(x)
        out2 = self.globalMaxPool(x)
        out = torch.squeeze(torch.cat((out1, out2), dim=1))
        return out


class DANN(torch.nn.Module):
    def __init__(self, architecture, pretrained=True, output_dim=1):
        super(DANN, self).__init__()
        assert architecture in ['shuffle05', 'shuffle10', 'mobile', 'resnet50'], 'wrong network architecture choice'
        if architecture == 'shuffle05':
            model = models.shufflenet_v2_x0_5(pretrained=pretrained)
            model.fc = torch.nn.Identity()
        elif architecture == 'shuffle10':
            model = models.shufflenet_v2_x1_0(pretrained=pretrained)
            model.fc = torch.nn.Identity()
        elif architecture == 'mobile':
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier = torch.nn.Identity()
        else:
            model = models.resnet50(pretrained=pretrained)
            model.fc = torch.nn.Identity()

        self.feature_extractor = model
        # summary(model.to('cuda'),(3,512,512))

        with torch.no_grad():
            a = self.feature_extractor.to('cpu')(torch.rand(2, 3, 512, 512))
        input_dim = a.view(2, -1).shape[1]

        self.label_classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128)
            , torch.nn.BatchNorm1d(128)
            , torch.nn.Dropout2d()
            , torch.nn.ReLU(True)
            , torch.nn.Linear(128, 64)
            , torch.nn.BatchNorm1d(64)
            , torch.nn.ReLU(True)
            , torch.nn.Linear(64, output_dim)
        )

        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128)
            , torch.nn.BatchNorm1d(128)
            , torch.nn.ReLU(True)
            , torch.nn.Linear(128, output_dim)
        )

        '''
        self.label_classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64)
            , torch.nn.BatchNorm1d(64)
            , torch.nn.ReLU(True)
            , torch.nn.Linear(64, output_dim)
        )

        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64)
            , torch.nn.BatchNorm1d(64)
            , torch.nn.ReLU(True)
            , torch.nn.Linear(64, 16)
            , torch.nn.BatchNorm1d(16)
            , torch.nn.ReLU(True)
            , torch.nn.Linear(16, 1)
        )
        '''

    def forward(self, x, alpha):
        feature = self.feature_extractor(x)
        feature = feature.view(feature.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_logit = self.label_classifier(feature)
        domain_logit = self.domain_classifier(reverse_feature)
        return class_logit, domain_logit


class FeatureExtractor(torch.nn.Module):
    def __init__(self, architecture, pretrained=True):
        super(FeatureExtractor, self).__init__()
        assert architecture in ['shuffle05', 'shuffle10', 'mobile', 'resnet50'], 'wrong network architecture choice'
        if architecture == 'shuffle05':
            model = models.shufflenet_v2_x0_5(pretrained=pretrained)
            model.fc = torch.nn.Identity()
        elif architecture == 'shuffle10':
            model = models.shufflenet_v2_x1_0(pretrained=pretrained)
            model.fc = torch.nn.Identity()
        elif architecture == 'mobile':
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier = torch.nn.Identity()
        else:
            model = models.resnet50(pretrained=pretrained)
            model.fc = torch.nn.Identity()
        self.feature_extractor = model

    def forward(self, x):
        feature = self.feature_extractor(x)
        return feature
        # return feature.view(feature.shape[0], -1)


class LabelClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LabelClassifier, self).__init__()
        self.lc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64)
            , torch.nn.BatchNorm1d(64)
            , torch.nn.ReLU(True)
            , torch.nn.Linear(64, output_dim)
        )

    def forward(self, feature):
        return self.lc(feature)


class DomainClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(DomainClassifier, self).__init__()
        self.dc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64)
            , torch.nn.BatchNorm1d(64)
            , torch.nn.ReLU(True)
            , torch.nn.Linear(64, 16)
            , torch.nn.BatchNorm1d(16)
            , torch.nn.ReLU(True)
            , torch.nn.Linear(16, 1)
        )

    def forward(self, feature):
        return self.dc(feature)

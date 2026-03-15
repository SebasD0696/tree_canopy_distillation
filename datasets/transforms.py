from torchvision import transforms

mae_transform = transforms.Compose([
    transforms.Resize((512,512)),   # MAE Teacher espera tamaño uniforme
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

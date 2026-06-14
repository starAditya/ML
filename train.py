    import warnings
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, random_split
    from sklearn.metrics import classification_report
    from tqdm import tqdm
    from PIL import ImageFile
    from torch.amp import GradScaler, autocast

    # =========================
    # SETTINGS
    # =========================

    warnings.filterwarnings("ignore")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    DATASET_PATH = r"C:\Users\gupta\.cache\kagglehub\datasets\pdavpoojan\the-rvlcdip-dataset-test\versions\1\test"

    BATCH_SIZE = 45
    EPOCHS = 5
    LEARNING_RATE = 1e-4

    torch.backends.cudnn.benchmark = True

    # =========================
    # SAFE DATASET
    # =========================

    class SafeImageFolder(ImageFolder):
        def __getitem__(self, index):
            try:
                return super().__getitem__(index)
            except Exception:
                print(f"\nSkipping corrupted file: {self.samples[index][0]}")
                return self.__getitem__((index + 1) % len(self.samples))

    # =========================
    # DEVICE
    # =========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("Using Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)

    print("=" * 50)

    # =========================
    # TRANSFORMS
    # =========================

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =========================
    # DATASET
    # =========================

    dataset = SafeImageFolder(
        root=DATASET_PATH,
        transform=transform
    )

    print("Classes:", dataset.classes)
    print("Total Images:", len(dataset))

    # =========================
    # SPLIT
    # =========================

    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    print("\nDataset Split:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # =========================
    # DATALOADERS
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================

    model = models.efficientnet_b0(weights="DEFAULT")

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(dataset.classes)
    )

    model = model.to(device)

    # =========================
    # LOSS / OPTIMIZER
    # =========================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = GradScaler("cuda")

    # =========================
    # EVALUATION
    # =========================

    def evaluate(loader):
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    # =========================
    # TRAINING
    # =========================

    best_val_acc = 0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = evaluate(train_loader)
        val_acc = evaluate(val_loader)

        print(
            f"\nEpoch {epoch+1}/{EPOCHS}"
            f" | Loss: {running_loss:.3f}"
            f" | Train Acc: {train_acc:.2f}%"
            f" | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "best_document_classifier.pth"
            )

            print("Best model saved!")

    # =========================
    # TEST
    # =========================

    print("\nEvaluating on Test Set...")

    model.load_state_dict(
        torch.load(
            "best_document_classifier.pth",
            map_location=device
        )
    )

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm(test_loader):

            images = images.to(device, non_blocking=True)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = (
        sum(
            p == l
            for p, l in zip(all_preds, all_labels)
        )
        / len(all_labels)
    ) * 100

    print("\n" + "=" * 50)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("=" * 50)

    print("\nClassification Report:\n")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=dataset.classes
        )
    )
def recall_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.float()
    y_pred = y_pred.float()
    tp = (y_true * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    recall = tp / (tp + fn + eps)
    return recall

def precision_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.float()
    y_pred = y_pred.float()
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    precision = tp / (tp + fp + eps)
    return precision

def dice_score(pred, target, epsilon=1e-6):


    # Flatten the tensors to 1D
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Compute Dice Score
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice

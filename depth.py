import numpy as np
from collections import Counter

def accuracy(node, data, target, predict_func):
    preds = predict_func(node, data)
    return np.mean(preds == data[target])

def post_prune(node, validation_data, target, predict_func):
    """
    انجام پس‌هرس روی درخت تصمیم با استفاده از داده اعتبارسنجی.
    """
    # اگر برگ است، هرس لازم نیست
    if node.is_leaf():
        return node

    # اول شاخه‌ها را به‌صورت بازگشتی هرس می‌کنیم
    for branch_value, child in node.branches.items():
        node.branches[branch_value] = post_prune(child,
                                                 validation_data[validation_data[node.feature] == branch_value],
                                                 target,
                                                 predict_func)

    # تست هرس این گره
    # پیش از هرس
    before_acc = accuracy(node, validation_data, target, predict_func)

    # اکثریت کلاس‌ها در این زیر درخت
    labels = validation_data[target]
    if labels.empty:
        return node  # داده‌ای برای تست نیست

    majority_label = Counter(labels).most_common(1)[0][0]
    leaf_node = type(node)(label=majority_label)  # ساخت برگ

    # بعد از هرس (یعنی جایگزینی با برگ)
    after_acc = accuracy(leaf_node, validation_data, target, predict_func)

    # اگر هرس باعث بهبود یا بدون تغییر دقت شد، گره را به برگ تبدیل می‌کنیم
    if after_acc >= before_acc:
        return leaf_node
    else:
        return node

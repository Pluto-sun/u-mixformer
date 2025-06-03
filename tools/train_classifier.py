import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mmseg.models.classifiers.umixformer_classifier import UMixFormerClassifier
from data_provider.data_factory import data_provider
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train UMixFormer Classifier')
    # 数据参数
    parser.add_argument('--data', type=str, default='SAHU', help='数据集名称')
    parser.add_argument('--root_path', type=str, default='./dataset/SAHU', help='数据根目录')
    parser.add_argument('--seq_len', type=int, default=64, help='时间窗口大小')
    parser.add_argument('--step', type=int, default=64, help='滑动窗口步长')
    parser.add_argument('--gaf_method', type=str, default='summation', help='GAF转换方法')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    
    # 模型参数
    parser.add_argument('--num_classes', type=int, default=3, help='分类类别数')
    parser.add_argument('--embed_dims', type=int, default=32, help='嵌入维度')
    parser.add_argument('--num_heads', type=list, default=[1, 2, 4], help='注意力头数')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--num_stages', type=int, default=3, help='网络阶段数')
    parser.add_argument('--num_layers', type=list, default=[2, 2, 2], help='每个阶段的层数')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    
    return parser.parse_args()

def setup_logging(args):
    """设置日志"""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    log_file = os.path.join(args.log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        # 调整数据维度顺序为 [B, C, H, W]
        data = data.permute(0, 3, 1, 2).to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    logger.info(f'Train Epoch: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, logger):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating'):
            data = data.permute(0, 3, 1, 2).to(device)
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    logger.info(f'Validation: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def main():
    args = parse_args()
    logger = setup_logging(args)
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 设置设备
    device = torch.device(args.device)
    logger.info(f'Using device: {device}')
    
    # 加载数据
    logger.info('Loading data...')
    train_dataset, train_loader = data_provider(args, 'train')
    val_dataset, val_loader = data_provider(args, 'val')
    test_dataset, test_loader = data_provider(args, 'test')
    
    # 获取输入通道数（特征维度）
    sample_data, _ = next(iter(train_loader))
    in_channels = sample_data.shape[-1]  # 最后一维是通道数
    logger.info(f'Input channels: {in_channels}')
    
    # 创建模型
    model = UMixFormerClassifier(
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=in_channels,
            embed_dims=args.embed_dims,
            num_stages=args.num_stages,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            patch_sizes=[4, 3, 3],
            sr_ratios=[4, 2, 1],
            out_indices=(0, 1, 2),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.drop_rate,
            drop_path_rate=args.drop_rate
        ),
        num_classes=args.num_classes,
        drop_rate=args.drop_rate
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # 训练循环
    logger.info('Starting training...')
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        logger.info(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, logger
        )
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, save_path)
            logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, save_path)
    
    # 测试最佳模型
    logger.info('Testing best model...')
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device, logger)
    logger.info(f'Final test accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    main() 
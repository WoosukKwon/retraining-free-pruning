class AverageMeter:

     def __init__(self, name):
         self.name = name
         self.reset()

     def reset(self):
         self.val = 0
         self.avg = 0
         self.sum = 0
         self.count = 0

     def update(self, val, n=1):
         self.val = val
         self.sum += val * n
         self.count += n
         self.avg = self.sum / self.count

     def __str__(self):
         return f"{self.name}: {self.avg:.4f}"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EquivariantSubsample(torch.nn.Module):
  def __init__(self, reduction=(2,2)):
    super().__init__()
    self.width_reduction = reduction[0]
    self.height_reduction = reduction[1]

  def max(self, sample):
    max_values, max_w_indices = torch.max(sample, dim=3)
    max_values, h = torch.max(max_values, dim=2)
    w = torch.gather(max_w_indices, dim=2, index=h.unsqueeze(-1)).squeeze(-1)
    max_indices = torch.stack((h, w), dim=-1)
    return max_values, max_indices

  def get_p(self, image):
    b, c, h, w = image.shape
    max_values, max_index = self.max(image)
    max_h_idx = max_index[::, ::, 0] % self.height_reduction
    max_w_idx = max_index[::, ::, 1] % self.width_reduction
    return max_h_idx, max_w_idx

  def block_pool(self, img, y_offsets, x_offsets):
    batch_size, channels, height, width = img.shape
    h = self.height_reduction
    w = self.width_reduction

    # Clamp to ensure offsets are within bounds for the pool_size
    y_offsets = torch.clamp(y_offsets, 0, height - h)
    x_offsets = torch.clamp(x_offsets, 0, width - w)

    # extend to batch and channel
    y_offsets = y_offsets.unsqueeze(-1).unsqueeze(-1).repeat(1,1,h,w).to(device)
    x_offsets = x_offsets.unsqueeze(-1).unsqueeze(-1).repeat(1,1,h,w).to(device)

    # Generate grid for h x w window
    y_relative = torch.arange(h).view(1, 1, h, 1).repeat(batch_size, channels, 1, w).to(device)
    x_relative = torch.arange(w).view(1, 1, 1, w).repeat(batch_size, channels, h,1).to(device)

    # Calculate absolute indices for y and x within the h x w block
    y_indices = y_offsets + y_relative
    x_indices = x_offsets + x_relative

    block = img[torch.arange(batch_size).view(-1, 1, 1, 1),
                torch.arange(channels).view(1, -1, 1, 1),
                y_indices, x_indices]
    block, indexes = self.max(block)
    return block.to(device)

  def forward(self, images, p):
    b, c, h, w = images.shape
    p_w = p[0].unsqueeze(-1).to(device)
    p_h = p[1].unsqueeze(-1).to(device)

    w_sample_indices = p_w + torch.arange(0, w, self.width_reduction).to(device)
    h_sample_indices = p_h + torch.arange(0, h, self.height_reduction).to(device)

    out_h = h//self.height_reduction
    out_w = w//self.width_reduction

    output = torch.zeros(b,c,out_h, out_w).to(device)
    for i in range(out_h):
      for j in range(out_w):
        y_offsets = h_sample_indices[::,::,i]
        x_offsets = w_sample_indices[::,::,j]
        output[::,::, i,j] = self.block_pool(images, y_offsets, x_offsets)
    return output
  
if __name__ == "__main__":
    #test equivariant function
    sub = EquivariantSubsample(reduction=(2,2))
    test_image = torch.tensor([[
                    [15,2,3,4,5],
                    [6,7,8,9,10],
                    [11,12,13,14,15],
                    [16,17,18,19,20],
                    [21,22,23,24,25]
                ],

                    [[1,2,3,4,5],
                    [6,7,8,9,10],
                    [11,12,13,14,15],
                    [16,17,2,25,20],
                    [21,22,22,24,23]],

                [[1,2,3,4,5],
                    [6,7,8,9,10],
                    [11,12,13,14,15],
                    [16,17,18,19,20],
                    [21,22,23,25,24]
                ]
                ]).repeat(4,1,1,1).to(device)
    print(test_image)
    print(test_image.shape)
    m, indices = sub.max(test_image)
    p = sub.get_p(test_image)
    print(f"max height: {p[0]}, max width: {p[1]}")
    print(m)
    print(indices)

    o = sub.forward(test_image, p)
    print(f"Outut: {o}")

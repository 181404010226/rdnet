import torch

def test_tensor_overflow():
    # Define large and small tensor values
    large_value1 = torch.tensor([1e7], dtype=torch.float16)
    large_value2 = torch.tensor([1e7], dtype=torch.float16)
    small_value1 = torch.tensor([1e-7], dtype=torch.float16)
    small_value2 = torch.tensor([1e-7], dtype=torch.float16)

    # Calculate the product of large values
    large_product = large_value1 * large_value2
    print(f"Product of large values: {large_product.item()}")

    # Calculate the product of small values
    small_product = small_value1 * small_value2
    print(f"Product of small values: {small_product.item()}")

# Run the test function
test_tensor_overflow()

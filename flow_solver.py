import torch
from typing import Callable

class FlowSolver:
    """
    Implements a simple ODE solver (Euler method) to 'flow' noise into an image.
    In Flow Matching, we move from t=0 (noise) to t=1 (data).
    """
    
    @torch.no_grad()
    def solve_euler(self, 
                    x_start: torch.Tensor, 
                    vector_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                    steps: int = 10) -> torch.Tensor:
        """
        Args:
            x_start: The starting point at t=0 (usually Gaussian noise).
            vector_field_fn: A function that takes (x, t) and returns the velocity.
                             This will eventually be our Neural Network.
            steps: Number of integration steps. More steps = higher quality but slower.
            
        Returns:
            The final state at t=1.
        """
        x = x_start
        dt = 1.0 / steps  # The size of each time step
        
        for i in range(steps):
            # Calculate current time t from 0 to 1
            t = torch.ones(x.shape[0], device=x.device) * (i / steps)
            
            # 1. Get the velocity (direction) from the vector field
            # In the future, this calls the model: model(x, t, landmarks)
            vt = vector_field_fn(x, t)
            
            # 2. Update x: x = x + vt * dt
            # This is the core 'Euler Step'
            x = x + vt * dt
            
        return x

# --- Learning Check: A Simple 2D Example ---
if __name__ == "__main__":
    # Let's simulate a simple vector field that pushes points 
    # from the origin (0,0) towards the point (5, 5).
    
    def simple_vector_field(x, t):
        # Velocity is simply the direction towards the target (5,5)
        target = torch.tensor([5.0, 5.0])
        return target - x

    solver = FlowSolver()
    
    # Start at origin
    start_pos = torch.tensor([[0.0, 0.0]])
    
    # Solve in 5 steps
    final_pos = solver.solve_euler(start_pos, simple_vector_field, steps=5)
    
    print(f"Starting at: {start_pos.numpy()}")
    print(f"Ending at:   {final_pos.numpy()} (Expected close to [5, 5])")

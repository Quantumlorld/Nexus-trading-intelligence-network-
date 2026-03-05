import { useMutation, useQuery } from '@tanstack/react-query';
import { authService } from '@/services/auth';
import { LoginRequest, User } from '@/types/api';
import { useAuthStore } from '@/store/authStore';
import { toast } from 'react-hot-toast';

export const useLogin = () => {
  const { login } = useAuthStore();

  return useMutation({
    mutationFn: (credentials: LoginRequest) => authService.login(credentials),
    onSuccess: (response) => {
      if (response.success && response.data) {
        login(response.data.user);
        toast.success('Login successful!');
      } else {
        toast.error(response.error || 'Login failed');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Login failed');
    },
  });
};

export const useLogout = () => {
  const { logout } = useAuthStore();

  return useMutation({
    mutationFn: () => authService.logout(),
    onSuccess: () => {
      logout();
      toast.success('Logged out successfully');
    },
    onError: (error: any) => {
      toast.error(error.message || 'Logout failed');
      // Force logout even if API call fails
      logout();
    },
  });
};

export const useProfile = () => {
  const { updateUser } = useAuthStore();

  return useQuery({
    queryKey: ['profile'],
    queryFn: () => authService.getProfile(),
    onSuccess: (response) => {
      if (response.success && response.data) {
        updateUser(response.data);
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load profile');
    },
  });
};

export const useUpdateProfile = () => {
  const { updateUser } = useAuthStore();

  return useMutation({
    mutationFn: (updates: { email?: string; password?: string }) =>
      authService.updateProfile(updates),
    onSuccess: (response) => {
      if (response.success && response.data) {
        updateUser(response.data);
        toast.success('Profile updated successfully!');
      } else {
        toast.error(response.error || 'Profile update failed');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Profile update failed');
    },
  });
};

export const useChangePassword = () => {
  return useMutation({
    mutationFn: (params: { current_password: string; new_password: string }) =>
      authService.changePassword(params),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Password changed successfully!');
      } else {
        toast.error(response.error || 'Password change failed');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Password change failed');
    },
  });
};

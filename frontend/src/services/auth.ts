import apiService from './api';
import { LoginRequest, LoginResponse, User, ApiResponse } from '@/types/api';

export class AuthService {
  // Login
  async login(credentials: LoginRequest): Promise<ApiResponse<LoginResponse>> {
    return apiService.login(credentials.username, credentials.password);
  }

  // Logout
  async logout(): Promise<ApiResponse<void>> {
    return apiService.logout();
  }

  // Get current user profile
  async getProfile(): Promise<ApiResponse<User>> {
    return apiService.get<User>('/auth/profile');
  }

  // Update profile
  async updateProfile(updates: {
    email?: string;
    password?: string;
  }): Promise<ApiResponse<User>> {
    return apiService.put<User>('/auth/profile', updates);
  }

  // Refresh token
  async refreshToken(): Promise<ApiResponse<{ access_token: string }>> {
    return apiService.post<{ access_token: string }>('/auth/refresh');
  }

  // Change password
  async changePassword(params: {
    current_password: string;
    new_password: string;
  }): Promise<ApiResponse<void>> {
    return apiService.post<void>('/auth/change-password', params);
  }

  // Request password reset
  async requestPasswordReset(email: string): Promise<ApiResponse<{ message: string }>> {
    return apiService.post<{ message: string }>('/auth/forgot-password', { email });
  }

  // Reset password
  async resetPassword(params: {
    token: string;
    new_password: string;
  }): Promise<ApiResponse<{ message: string }>> {
    return apiService.post<{ message: string }>('/auth/reset-password', params);
  }

  // Enable 2FA
  async enable2FA(): Promise<ApiResponse<{ qr_code: string; secret: string }>> {
    return apiService.post<{ qr_code: string; secret }>('/auth/2fa/enable');
  }

  // Disable 2FA
  async disable2FA(code: string): Promise<ApiResponse<{ message: string }>> {
    return apiService.post<{ message: string }>('/auth/2fa/disable', { code });
  }

  // Verify 2FA
  async verify2FA(code: string): Promise<ApiResponse<{ verified: boolean }>> {
    return apiService.post<{ verified: boolean }>('/auth/2fa/verify', { code });
  }
}

export const authService = new AuthService();

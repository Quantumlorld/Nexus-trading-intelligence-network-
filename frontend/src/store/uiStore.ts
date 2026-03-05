import { create } from 'zustand';

interface UIState {
  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;

  // Theme
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;

  // Notifications
  notificationsOpen: boolean;
  toggleNotifications: () => void;
  setNotificationsOpen: (open: boolean) => void;

  // Modals
  modals: {
    tradeForm: boolean;
    confirmTrade: boolean;
    reconcileModal: boolean;
    settings: boolean;
  };
  openModal: (modal: keyof UIState['modals']) => void;
  closeModal: (modal: keyof UIState['modals']) => void;
  closeAllModals: () => void;

  // Loading states
  loading: {
    positions: boolean;
    trades: boolean;
    balance: boolean;
    reconciliation: boolean;
  };
  setLoading: (key: keyof UIState['loading'], value: boolean) => void;

  // Error handling
  error: string | null;
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  // Sidebar
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open: boolean) => set({ sidebarOpen: open }),

  // Theme
  theme: 'light',
  toggleTheme: () => set((state) => ({ theme: state.theme === 'light' ? 'dark' : 'light' })),
  setTheme: (theme: 'light' | 'dark') => set({ theme }),

  // Notifications
  notificationsOpen: false,
  toggleNotifications: () => set((state) => ({ notificationsOpen: !state.notificationsOpen })),
  setNotificationsOpen: (open: boolean) => set({ notificationsOpen: open }),

  // Modals
  modals: {
    tradeForm: false,
    confirmTrade: false,
    reconcileModal: false,
    settings: false,
  },
  openModal: (modal) =>
    set((state) => ({
      modals: { ...state.modals, [modal]: true },
    })),
  closeModal: (modal) =>
    set((state) => ({
      modals: { ...state.modals, [modal]: false },
    })),
  closeAllModals: () =>
    set({
      modals: {
        tradeForm: false,
        confirmTrade: false,
        reconcileModal: false,
        settings: false,
      },
    }),

  // Loading states
  loading: {
    positions: false,
    trades: false,
    balance: false,
    reconciliation: false,
  },
  setLoading: (key, value) =>
    set((state) => ({
      loading: { ...state.loading, [key]: value },
    })),

  // Error handling
  error: null,
  setError: (error) => set({ error }),
  clearError: () => set({ error: null }),
}));

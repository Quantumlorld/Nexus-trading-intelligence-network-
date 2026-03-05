import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useMutation } from '@tanstack/react-query';
import { tradingService } from '@/services/trading';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useUIStore } from '@/store/uiStore';
import { toast } from 'react-hot-toast';

const tradeSchema = z.object({
  symbol: z.string().min(1, 'Symbol is required'),
  trade_type: z.enum(['BUY', 'SELL']),
  quantity: z.number().positive('Quantity must be positive'),
  order_type: z.enum(['MARKET', 'LIMIT', 'STOP']),
  price: z.number().positive('Price must be positive').optional(),
  stop_loss: z.number().positive('Stop loss must be positive').optional(),
  take_profit: z.number().positive('Take profit must be positive').optional(),
  leverage: z.number().positive('Leverage must be positive').default(1),
});

type TradeFormData = z.infer<typeof tradeSchema>;

const TradeForm: React.FC = () => {
  const { closeModal } = useUIStore();
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false);;

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<TradeFormData>({
    resolver: zodResolver(tradeSchema),
    defaultValues: {
      leverage: 1,
      trade_type: 'BUY',
      order_type: 'MARKET',
    },
  });

  const orderType = watch('order_type');
  const tradeType = watch('trade_type');

  const executeTrade = useMutation({
    mutationFn: (data: TradeFormData) => tradingService.executeTrade(data),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Trade executed successfully!');
        closeModal('tradeForm');
      } else {
        toast.error(response.error || 'Trade execution failed');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Trade execution failed');
    },
  });

  const onSubmit = (data: TradeFormData) => {
    if (isSubmitting) return;
    setIsSubmitting(true);
    // Validate required fields based on order type
    if (orderType === 'LIMIT' && !data.price) {
      toast.error('Price is required for limit orders');
      return;
    }
    if (orderType === 'STOP' && !data.price) {
      toast.error('Price is required for stop orders');
      return;
    }

    try {
      executeTrade.mutate(data);
    } finally {
      setIsSubmitting(false);
    }
  };

  const calculateRisk = () => {
    const quantity = watch('quantity') || 0;
    const stopLoss = watch('stop_loss');
    const entryPrice = watch('price') || 0;

    if (quantity && stopLoss && entryPrice) {
      const riskPerUnit = Math.abs(entryPrice - stopLoss);
      const totalRisk = riskPerUnit * quantity;
      return totalRisk;
    }
    return 0;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-gray-900">New Trade</h2>
          <Button
            variant="ghost"
            onClick={() => closeModal('tradeForm')}
            className="p-2"
          >
            ×
          </Button>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {/* Symbol */}
          <Input
            label="Symbol"
            placeholder="e.g., EUR/USD"
            error={errors.symbol?.message}
            {...register('symbol')}
          />

          {/* Trade Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Trade Type
            </label>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant={tradeType === 'BUY' ? 'primary' : 'outline'}
                onClick={() => setValue('trade_type', 'BUY')}
                className={tradeType === 'BUY' ? 'bg-green-600 hover:bg-green-700' : ''}
              >
                BUY
              </Button>
              <Button
                type="button"
                variant={tradeType === 'SELL' ? 'primary' : 'outline'}
                onClick={() => setValue('trade_type', 'SELL')}
                className={tradeType === 'SELL' ? 'bg-red-600 hover:bg-red-700' : ''}
              >
                SELL
              </Button>
            </div>
          </div>

          {/* Quantity */}
          <Input
            label="Quantity"
            type="number"
            step="0.01"
            placeholder="0.01"
            error={errors.quantity?.message}
            {...register('quantity', { valueAsNumber: true })}
          />

          {/* Order Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Order Type
            </label>
            <select
              {...register('order_type')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="MARKET">Market</option>
              <option value="LIMIT">Limit</option>
              <option value="STOP">Stop</option>
            </select>
          </div>

          {/* Price (for LIMIT/STOP orders) */}
          {(orderType === 'LIMIT' || orderType === 'STOP') && (
            <Input
              label={orderType === 'LIMIT' ? 'Limit Price' : 'Stop Price'}
              type="number"
              step="0.00001"
              placeholder="1.00000"
              error={errors.price?.message}
              {...register('price', { valueAsNumber: true })}
            />
          )}

          {/* Advanced Options Toggle */}
          <Button
            type="button"
            variant="ghost"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full"
          >
            {showAdvanced ? 'Hide' : 'Show'} Advanced Options
          </Button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-4 border-t pt-4">
              {/* Stop Loss */}
              <Input
                label="Stop Loss"
                type="number"
                step="0.00001"
                placeholder="0.99000"
                error={errors.stop_loss?.message}
                {...register('stop_loss', { valueAsNumber: true })}
              />

              {/* Take Profit */}
              <Input
                label="Take Profit"
                type="number"
                step="0.00001"
                placeholder="1.01000"
                error={errors.take_profit?.message}
                {...register('take_profit', { valueAsNumber: true })}
              />

              {/* Leverage */}
              <Input
                label="Leverage"
                type="number"
                step="1"
                placeholder="1"
                error={errors.leverage?.message}
                {...register('leverage', { valueAsNumber: true })}
              />

              {/* Risk Calculation */}
              {calculateRisk() > 0 && (
                <div className="bg-gray-50 p-3 rounded-md">
                  <div className="text-sm text-gray-600">
                    <div>Potential Risk: <span className="font-medium">${calculateRisk().toFixed(2)}</span></div>
                    <div>Risk per Unit: <span className="font-medium">${Math.abs((watch('price') || 0) - (watch('stop_loss') || 0)).toFixed(5)}</span></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Submit Buttons */}
          <div className="flex gap-3 pt-4">
            <Button
              type="submit"
              className="flex-1"
              loading={executeTrade.isPending}
              disabled={executeTrade.isPending || isSubmitting}
            >
              {executeTrade.isPending ? 'Executing...' : 'Execute Trade'}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => closeModal('tradeForm')}
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default TradeForm;

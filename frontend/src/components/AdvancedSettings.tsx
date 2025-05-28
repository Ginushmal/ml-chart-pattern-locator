import React, { useState } from 'react';

interface AdvancedSettingsProps {
    onSettingsChange: (settings: PatternDetectionSettings) => void;
}

export interface PatternDetectionSettings {
    min_window_size: number;
    max_window_size: number;
    padding_proportion: number;
    stride: number;
    prob_threshold_of_no_pattern: number;
    min_eps: number;
    max_eps: number;
    min_samples: number;
    iou_threshold: number;
    probability_difference_threshold: number;
    final_avg_probability_threshold: number;
    pattern_thresholds: {
        double_top: number;
        double_bottom: number;
        triangle_symmetrical: number;
        head_and_shoulders_top: number;
        head_and_shoulders_bottom: number;
        flag_high_and_tight: number;
    };
}

const defaultSettings: PatternDetectionSettings = {
    min_window_size: 10,
    max_window_size: 50,
    padding_proportion: 0.6,
    stride: 1,
    prob_threshold_of_no_pattern: 0.5,
    min_eps: 0.02,
    max_eps: 0.1,
    min_samples: 3,
    iou_threshold: 0.4,
    probability_difference_threshold: 0.1,
    final_avg_probability_threshold: 0.85,
    pattern_thresholds: {
        double_top: 0.9,
        double_bottom: 0.9,
        triangle_symmetrical: 0.4309,
        head_and_shoulders_top: 0.4583,
        head_and_shoulders_bottom: 0.3961,
        flag_high_and_tight: 0.5506
    }
};

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({ onSettingsChange }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [isPatternThresholdsOpen, setIsPatternThresholdsOpen] = useState(false);
    const [settings, setSettings] = useState<PatternDetectionSettings>(defaultSettings);

    const handleChange = (key: keyof PatternDetectionSettings, value: string) => {
        const numValue = parseFloat(value);
        if (!isNaN(numValue)) {
            const newSettings = { ...settings, [key]: numValue };
            setSettings(newSettings);
            onSettingsChange(newSettings);
        }
    };

    const handlePatternThresholdChange = (pattern: keyof PatternDetectionSettings['pattern_thresholds'], value: string) => {
        const numValue = parseFloat(value);
        if (!isNaN(numValue)) {
            const newSettings = {
                ...settings,
                pattern_thresholds: {
                    ...settings.pattern_thresholds,
                    [pattern]: numValue
                }
            };
            setSettings(newSettings);
            onSettingsChange(newSettings);
        }
    };

    return (
        <div className="bg-gray-800 p-4 rounded-lg shadow-lg mb-4">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex justify-between items-center text-white font-semibold"
            >
                <span>Advanced Settings</span>
                <span>{isOpen ? '▼' : '▶'}</span>
            </button>

            {isOpen && (
                <div className="mt-4 space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-300">Min Window Size</label>
                            <input
                                type="number"
                                value={settings.min_window_size}
                                onChange={(e) => handleChange('min_window_size', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                min="1"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">Max Window Size</label>
                            <input
                                type="number"
                                value={settings.max_window_size}
                                onChange={(e) => handleChange('max_window_size', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                min="1"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">Padding Proportion</label>
                            <input
                                type="number"
                                value={settings.padding_proportion}
                                onChange={(e) => handleChange('padding_proportion', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                step="0.1"
                                min="0"
                                max="1"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">Stride</label>
                            <input
                                type="number"
                                value={settings.stride}
                                onChange={(e) => handleChange('stride', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                min="1"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">No Pattern Threshold</label>
                            <input
                                type="number"
                                value={settings.prob_threshold_of_no_pattern}
                                onChange={(e) => handleChange('prob_threshold_of_no_pattern', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                step="0.1"
                                min="0"
                                max="1"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">Min Epsilon</label>
                            <input
                                type="number"
                                value={settings.min_eps}
                                onChange={(e) => handleChange('min_eps', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                step="0.01"
                                min="0"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">Max Epsilon</label>
                            <input
                                type="number"
                                value={settings.max_eps}
                                onChange={(e) => handleChange('max_eps', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                step="0.01"
                                min="0"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">Min Samples</label>
                            <input
                                type="number"
                                value={settings.min_samples}
                                onChange={(e) => handleChange('min_samples', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                min="1"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">IoU Threshold</label>
                            <input
                                type="number"
                                value={settings.iou_threshold}
                                onChange={(e) => handleChange('iou_threshold', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                step="0.1"
                                min="0"
                                max="1"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">Probability Difference Threshold</label>
                            <input
                                type="number"
                                value={settings.probability_difference_threshold}
                                onChange={(e) => handleChange('probability_difference_threshold', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                step="0.1"
                                min="0"
                                max="1"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300">Final Avg Probability Threshold</label>
                            <input
                                type="number"
                                value={settings.final_avg_probability_threshold}
                                onChange={(e) => handleChange('final_avg_probability_threshold', e.target.value)}
                                className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                step="0.1"
                                min="0"
                                max="1"
                            />
                        </div>
                    </div>

                    {/* Pattern-specific thresholds section */}
                    <div className="mt-6">
                        <button
                            onClick={() => setIsPatternThresholdsOpen(!isPatternThresholdsOpen)}
                            className="w-full flex justify-between items-center text-white font-semibold bg-gray-700 p-2 rounded-md"
                        >
                            <span>Pattern-specific Probability Thresholds</span>
                            <span>{isPatternThresholdsOpen ? '▼' : '▶'}</span>
                        </button>

                        {isPatternThresholdsOpen && (
                            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-300">Double Top</label>
                                    <input
                                        type="number"
                                        value={settings.pattern_thresholds.double_top}
                                        onChange={(e) => handlePatternThresholdChange('double_top', e.target.value)}
                                        className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                        step="0.0001"
                                        min="0"
                                        max="1"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-300">Double Bottom</label>
                                    <input
                                        type="number"
                                        value={settings.pattern_thresholds.double_bottom}
                                        onChange={(e) => handlePatternThresholdChange('double_bottom', e.target.value)}
                                        className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                        step="0.0001"
                                        min="0"
                                        max="1"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-300">Triangle (Symmetrical)</label>
                                    <input
                                        type="number"
                                        value={settings.pattern_thresholds.triangle_symmetrical}
                                        onChange={(e) => handlePatternThresholdChange('triangle_symmetrical', e.target.value)}
                                        className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                        step="0.0001"
                                        min="0"
                                        max="1"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-300">Head and Shoulders Top</label>
                                    <input
                                        type="number"
                                        value={settings.pattern_thresholds.head_and_shoulders_top}
                                        onChange={(e) => handlePatternThresholdChange('head_and_shoulders_top', e.target.value)}
                                        className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                        step="0.0001"
                                        min="0"
                                        max="1"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-300">Head and Shoulders Bottom</label>
                                    <input
                                        type="number"
                                        value={settings.pattern_thresholds.head_and_shoulders_bottom}
                                        onChange={(e) => handlePatternThresholdChange('head_and_shoulders_bottom', e.target.value)}
                                        className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                        step="0.0001"
                                        min="0"
                                        max="1"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-300">Flag (High and Tight)</label>
                                    <input
                                        type="number"
                                        value={settings.pattern_thresholds.flag_high_and_tight}
                                        onChange={(e) => handlePatternThresholdChange('flag_high_and_tight', e.target.value)}
                                        className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 text-white"
                                        step="0.0001"
                                        min="0"
                                        max="1"
                                    />
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default AdvancedSettings; 
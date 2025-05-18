from rest_framework import serializers

class PatternRequestSerializer(serializers.Serializer):
    symbol = serializers.CharField(max_length=10)
    start_date = serializers.DateField(format='%Y-%m-%d')
    end_date = serializers.DateField(format='%Y-%m-%d')

class PatternResponseSerializer(serializers.Serializer):
    chart_pattern = serializers.CharField(source='Chart Pattern')
    cluster = serializers.IntegerField(source='Cluster')
    start = serializers.CharField(source='Start')
    end = serializers.CharField(source='End')
    seg_start = serializers.CharField(source='Seg_Start')
    seg_end = serializers.CharField(source='Seg_End')
    avg_probability = serializers.FloatField(source='Avg_Probability')
    calc_start = serializers.CharField(source='Calc_Start')
    calc_end = serializers.CharField(source='Calc_End')
    window_size = serializers.IntegerField(source='Window_Size') 
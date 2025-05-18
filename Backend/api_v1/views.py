from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PatternRequestSerializer, PatternResponseSerializer
import yfinance as yf
import pandas as pd
from datetime import datetime
from utils.patternLocating import locate_patterns
import traceback

# Create your views here.

class PatternDetectionView(APIView):
    def post(self, request):
        # Validate input
        serializer = PatternRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Get validated data
        symbol = serializer.validated_data['symbol']
        start_date = serializer.validated_data['start_date']
        end_date = serializer.validated_data['end_date']
        
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return Response(
                    {"error": "No data found for the given symbol and date range"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Ensure 'Date' is datetime and convert to naive UTC
            df['Date'] = pd.to_datetime(df['Date'])
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
            # If it was already naive, we assume it's effectively UTC or consistently handled.
            
            # Call pattern detection function, disabling plotting for API calls
            patterns_df = locate_patterns(df, plot_count=0)
            
            if patterns_df is None or patterns_df.empty:
                return Response(
                    {"message": "No patterns found for the given data."},
                    status=status.HTTP_200_OK
                )
            
            # Convert datetime columns to string format for serialization
            # (These should now be naive from locate_patterns if derived from the naive UTC Date input)
            datetime_columns = ['Start', 'End', 'Seg_Start', 'Seg_End', 'Calc_Start', 'Calc_End']
            for col in datetime_columns:
                if col in patterns_df.columns:
                    # Ensure the column is actually datetime before trying to format
                    if pd.api.types.is_datetime64_any_dtype(patterns_df[col]):
                        patterns_df[col] = pd.to_datetime(patterns_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    elif not patterns_df[col].empty and isinstance(patterns_df[col].iloc[0], str):
                        pass # Already string, do nothing
                    else:
                        # Handle cases where it might be other types or mixed, convert to string safely
                        patterns_df[col] = patterns_df[col].astype(str)
            
            # Convert to dictionary format
            patterns_dict = patterns_df.to_dict('records')
            
            # Serialize the results
            response_serializer = PatternResponseSerializer(patterns_dict, many=True)
            
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            error_message = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_message)  # This will show in your Django console
            return Response(
                {"error": str(e), "detail": error_message},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

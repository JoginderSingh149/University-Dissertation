from django.db import models

class BookingData(models.Model):
    check_in = models.DateTimeField()
   # check_out = models.DateTimeField(null=True, blank=True)  # Optional based on your data
    nights = models.FloatField()
    guest = models.CharField(max_length=255) #models.IntegerField() #String
    total = models.FloatField()
    number_of_bookings = models.IntegerField(default=1) 
    # add here booking num 
    # add here data for kpis
    class Meta:
        unique_together = (('check_in', 'nights', 'total'),) #guest name/ add in booking number or both   # Adjust based on what defines uniqueness in your data

class OptimizationParameters(models.Model):
    p = models.IntegerField()
    d = models.IntegerField()
    q = models.IntegerField()
    seasonal_p = models.IntegerField()
    seasonal_d = models.IntegerField()
    seasonal_q = models.IntegerField()
    s = models.IntegerField(default=12)  # Assuming a constant seasonal period
    last_updated = models.DateTimeField(auto_now=True)

from django.db import models

class ProphetOptimizationParameters(models.Model):
    changepoint_prior_scale = models.FloatField()
    seasonality_prior_scale = models.FloatField()
    yearly_seasonality = models.BooleanField()
    rmse = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)  # Automatically updates to current timestamp when the object is saved

    class Meta:
        verbose_name = "Prophet Optimization Parameter"
        verbose_name_plural = "Prophet Optimization Parameters"

    def __str__(self):
        return f"Prophet Params Updated: {self.last_updated}"

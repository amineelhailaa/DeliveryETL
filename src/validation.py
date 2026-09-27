def validate_age(age):
    valid = age.notna() & (age > 0) & (age % 1 == 0)
    return age.where(valid)


def validate_ratings(ratings):
    return ratings.where(ratings.between(0, 5))



def validate_vehicle_condition(condition):
    return condition.where(condition.isin([0, 1, 2, 3]))



def validate_multiple_deliveries(deliveries):
    valid = deliveries.notna() & (deliveries >= 0) & (deliveries % 1 == 0)
    return deliveries.where(valid)



def validate_weather(weather):
    return weather.where(weather.notna())



def validate_vehicle_type(vehicle_type):
    return vehicle_type.where(vehicle_type.notna())



def validate_order_type(order_type):
    return order_type.where(order_type.isin(["Snack", "Drinks", "Buffet", "Meal"]))



def validate_festival(festival):
    return festival.where(festival.isin(["Yes", "No"]))


def validate_city(city):
    return city.where(city.isin(["Urban", "Metropolitian", "Semi-Urban"]))



def validate_traffic(traffic):
    return traffic.where(traffic.isin(["Low", "Medium", "High", "Jam"]))



def validate_distance(distance):
    return distance.where(distance.notna() & (distance >= 0))



def validate_order_hour(hour):
    return hour.where(hour.between(0, 23) & (hour % 1 == 0))

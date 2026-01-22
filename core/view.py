from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views import View
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

class HomePageView(View):
    def get(self, request):
        return render(request, 'home.html')




from django.views.decorators.csrf import csrf_exempt


from .shared_state import get_state


@csrf_exempt
def getdata(request):
    """
    API endpoint - monitoring statistikasini qaytaradi
    
    URL: /getdata/
    Method: GET
    Response: {"barcha": int, "faol": int, "uxlayotgan": int}
    """
    state = get_state()
    return JsonResponse(state)
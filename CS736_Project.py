from __future__ import division
from queue import Queue
import cv2
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from math import exp, pow

def bfs(Graph, tot_nodes, s, t, parent):
  q=Queue()
  visited_vertices=np.zeros(tot_nodes)
  q.put(s)
  visited_vertices[s]=True
  parent[s]=-1

  while not q.empty():
    current=q.get()
    for next in range(tot_nodes):
      if(visited_vertices[next]!=True and Graph[current][next]>0):
        q.put(next)
        parent[next]=current
        visited_vertices[next]= True
  return visited_vertices[t]


def dfs(rGraph, V, s, visited):
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if rGraph[v][u]])

def augmentingPath(Graph,source,sink):
  print("Augmenting path algorithm started")
  copy=Graph.copy()
  tot_nodes=copy.shape[0]
  parent=np.zeros(tot_nodes,dtype='int32')

  while bfs(copy,tot_nodes,source,sink,parent):
    bottleneck=float("inf")
    vertex=sink
    while(vertex!=source):
      bottleneck=min(bottleneck,copy[parent[vertex]][vertex])
      vertex=parent[vertex]
    #We have now calculated the bottleneck of a connected path between the source and sink and must update the residual graph now
    vertex=sink
    while vertex!=source:
      copy[parent[vertex]][vertex]-= bottleneck
      copy[vertex][parent[vertex]]+= bottleneck
      vertex=parent[vertex]

  s_u_path=np.zeros(tot_nodes)
  dfs(copy,tot_nodes,source,s_u_path)

  cuts=[]

  for u in range(tot_nodes):
    for v in range(tot_nodes):
      if s_u_path[u] and not s_u_path[v] and Graph[u][v]:
        cuts.append((u,v))

  return cuts

graphCutAlgo = {"ap": augmentingPath}
SIGMA = 30
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 10
LOADSEEDS = False

def edgeweights(u,v):
  return (SIGMA*SIGMA)*exp(-pow((int)(u)-(int)(v),2)/(2*pow(SIGMA,2)))

def edge_neighbours(graph,image):
  maxedge = -float("inf")
  r=image.shape[0]
  c=image.shape[1]
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      u = i * c + j
      if i + 1 < r: # pixel below
                v = (i + 1) * c + j
                edge = edgeweights(image[i][j], image[i + 1][j])
                graph[u][v] = max(edge+0.001*((int)(image[i][j])-(int)(image[i+1][j])),0) 
                graph[v][u]=max(edge+0.001*(-(int)(image[i][j])+(int)(image[i+1][j])),0)
                maxedge = max(maxedge, edge)
      if j + 1 < c: # pixel to the right
                v = i * c + j + 1
                edge = edgeweights(image[i][j], image[i][j + 1])
                graph[u][v] = max(edge+0.001*((int)(image[i][j])-(int)(image[i][j+1])),0)
                graph[v][u]=max(edge+0.001*(-(int)(image[i][j])+(int)(image[i][j+1])),0)
                maxedge = max(maxedge, edge)
  return maxedge
  
def average(matrix):
    sum=0
    count=0
    K = -float("inf")
    r, c = matrix.shape
    for i in range(r):
        for j in range(c):
            if i + 1 < r: # pixel below
                count=count+1
                sum+=abs((int)(matrix[i][j])-(int)(matrix[i+1][j]))
           
            if j + 1 < c: # pixel to the right
                count=count+1
                sum+=abs((int)(matrix[i][j])-(int)(matrix[i][j+1]))
    return sum/count

def plantSeed(image):

    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print ("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while (1):
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()


    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 10
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False


    paintSeeds(OBJ)
    paintSeeds(BKG)
    return seeds, image

def seeded_edges(graph,seeds,maxedge):
  r=seeds.shape[0]
  c=seeds.shape[1]
  for i in range(seeds.shape[0]):
    for j in range(seeds.shape[1]):
      u = i * c + j
      if seeds[i][j]==OBJCODE:
        graph[SOURCE][u] = maxedge
      elif seeds[i][j] == BKGCODE:
        graph[u][SINK] = maxedge

def imageToGraph(image):
  tot_nodes=image.size+2
  graph=np.zeros((tot_nodes,tot_nodes),dtype='int32')
  maxedge = edge_neighbours(graph, image)
  seeds, seededImage = plantSeed(image)
  seeded_edges(graph, seeds, maxedge)
  return graph, seededImage

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image

def imageSegmentation(imagefile, size=(50, 50), algo="ff"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    SIGMA=average(image)
    graph, seededImage = imageToGraph(image)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)
    #avg= average_non_zero(matrix)
    #print(avg)
    global SOURCE, SINK
    SOURCE += len(graph)
    SINK   += len(graph)

    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)
    print ("cuts:")
    print( cuts)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print ("Saved image as", savename)

def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s",
                        default=30, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    imageSegmentation(args.imagefile, (args.size, args.size), args.algo)
    
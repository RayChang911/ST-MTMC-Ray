import dijkstra
import cv2
import json
import os
import numpy as np
from rtree import index

from CONFIGS import *
from utils.utils import distance

class RouteNetwork():
    def __init__(self, loadPath: str=os.path.join(ROOT_PATH, 'routeNet.json')):
        with open(loadPath, 'r+') as f:
            self.nodes = json.load(f)
        
        self.graph = dijkstra.Graph()
        self.idx = index.Index()
        for nid, node in self.nodes.items():
            self.idx.insert(int(nid), (
                node['coordinate'][0], 
                node['coordinate'][1], 
                node['coordinate'][0], 
                node['coordinate'][1]
            ))
            for n in node['adjacents']:
                self.graph.add_edge(nid, n, float(node['adjacents'][n]))

    def NDistance(self, start_pt: list, end_pt: list) -> tuple[list[str], float]:
        d = 0.
        start_node = list(self.idx.nearest(
            (start_pt[0], start_pt[1], start_pt[0], start_pt[1]), 1))[0]
        end_node = list(self.idx.nearest(
            (end_pt[0], end_pt[1], end_pt[0], end_pt[1]), 1))[0]
        start_node = str(start_node).zfill(2)
        end_node = str(end_node).zfill(2)
        tmp_dijkstra = dijkstra.DijkstraSPF(self.graph, str(start_node))
        path = tmp_dijkstra.get_path(str(end_node))
        if len(path) > 2:
            # d += np.linalg.norm(start_pt -
            #                     RouteNetwork.nodes[str(start_node)]['coordinate'])
            for i, p in enumerate(path[:-1]):
                d += float(self.nodes[str(p)]['adjacents'][str(path[i+1])])
            # d += np.linalg.norm(start_pt -
            #                     RouteNetwork.nodes[str(end_node)]['coordinate'])
        else:
            d += np.linalg.norm(np.asarray(start_pt) - np.asarray(end_pt))
        return path, d
        
    def asImg(self):
        canvas = cv2.imread(os.path.join(ROOT_PATH,'FloorPlan.jpg'))
        for i, node in self.nodes.items():
            cv2.circle(
                canvas, 
                tuple(node['coordinate']), 
                10, 
                (0, 255, 0), 
                -1
            )
            cv2.putText(
                canvas, 
                i, 
                tuple(node['coordinate']),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 0)
            )
            for n in node['adjacents']:
                cv2.arrowedLine(
                    canvas, 
                    tuple(node['coordinate']), 
                    tuple(self.nodes[n]['coordinate']), 
                    (0, 255, 0), 
                    2
                )
        return canvas

def makeRouteNetWork():
    def create_node(x, y):
        global canvas, nodes
        n = input('Please input node id: ')
        while n in list(nodes.keys()):
            print('Node {} already exist! ')
            n = input('Please input node id: ')
        n = str(n)
        new = {}
        new['coordinate'] = (x, y)
        adjacent_nodes = {}
        while True:
            inp = ''
            inp = input('Please input adjacent node, x to stop: ')
            if inp != 'x':
                if inp not in list(nodes.keys()):
                    print('Node {} is not exist! '.format(inp))
                    continue
                d = distance((x, y), nodes[inp]['coordinate'])
                adjacent_nodes[inp] = d
                nodes[inp]['adjacents'][n] = d
                cv2.arrowedLine(
                    canvas, (x, y), tuple(nodes[inp]['coordinate']), (0, 255, 0), 2)
                cv2.arrowedLine(
                    canvas, tuple(nodes[inp]['coordinate']), (x, y), (0, 255, 0), 2)
            else:
                break
        new['adjacents'] = adjacent_nodes
        seen_cameras = []
        while True:
            inp = ''
            inp = input('Please input seen camera, x to stop: ')
            if inp != 'x':
                seen_cameras.append(inp)
            else:
                break
        new['cameras'] = seen_cameras
        cv2.circle(canvas, (x, y), 10, (0, 255, 0), -1)
        cv2.putText(canvas, n, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        nodes[n] = new
        # print(nodes)
        cv2.imshow('Floor Plan', canvas)
        with open(os.path.join(ROOT_PATH,'routeNet.json'), 'w') as f:
            json.dump(nodes, f)


    def getposition(event, x, y, flags, param):
        global ix, iy, nodes
        if event == cv2.EVENT_LBUTTONDBLCLK:
            ix, iy = x, y
            tmp_canvas = canvas.copy()
            cv2.circle(tmp_canvas, (x, y), 10, (0, 0, 255), -1)
            print('The image coordinate is {}, {}'.format(ix, iy))
            cv2.imshow('Floor Plan', tmp_canvas)
            cv2.waitKey(1)
            create_node(ix, iy)

    global canvas, nodes
    try:
        with open(os.path.join(ROOT_PATH,'routeNet.json'), 'r+') as f:
            nodes = json.load(f)
    except FileNotFoundError:
        nodes = {}
        pass

    plan = cv2.imread(os.path.join(ROOT_PATH,'FloorPlan.jpg'))
    canvas = plan.copy()
    cv2.namedWindow('Floor Plan')
    cv2.setMouseCallback('Floor Plan', getposition)

    """
    Draw Existed Nodes
    """
    for i, node in nodes.items():
        cv2.circle(canvas, tuple(node['coordinate']), 10, (0, 255, 0), -1)
        cv2.putText(canvas, i, tuple(node['coordinate']),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        for a in node['adjacents']:
            cv2.arrowedLine(
                canvas, tuple(node['coordinate']), tuple(nodes[a]['coordinate']), (0, 255, 0), 2)

    cv2.imshow('Floor Plan', canvas)
    cv2.waitKey(0)

if __name__ == '__main__':
    makeRouteNetWork()
    '''
    RouteNet = RouteNetwork()
    cv2.imshow('test route net', RouteNet.r())
    cv2.waitKey(0)
    '''
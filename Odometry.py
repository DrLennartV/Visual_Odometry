from glob import glob
import cv2, os, skimage
import numpy as np
import math

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        self.constant = 1e-6
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))

        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
        
        self.k = np.array([[self.focal_length, 0.000000000000e+00, self.pp[0]],
                    [0.000000000000e+00, self.focal_length, self.pp[1]],
                    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    
    def fastfeature(self, img):
        detector = cv2.FastFeatureDetector_create(threshold=29, nonmaxSuppression=True)
        m = detector.detect(img)
        return np.array([x.pt for x in m], dtype=np.float32).reshape(-1, 1, 2)
    
    def track(self, img1, img2, feature):
        re, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, feature, None)
        re = re[st == 1]
        feature = feature[st == 1]
        return re, feature
   
    '''
    def essentialMat_8_points(self, pts1, pts2):
        num1 = np.ones(pts1.shape[0]).reshape(pts1.shape[0], 1)
        pts1 = np.hstack((pts1, num1)).T
        pts2 = np.hstack((pts2, num1)).T
        avg1 = np.mean(pts1[:2], axis=1)
        len = pts1.shape[1]
        mul1 = np.array([[np.sqrt(2) / np.std(pts1[:2]), 0, -np.sqrt(2) / np.std(pts1[:2]) * avg1[0]],
                   [0, np.sqrt(2) / np.std(pts1[:2]), -np.sqrt(2) / np.std(pts1[:2]) * avg1[1]], 
                   [0, 0, 1]])
        
        avg2 = np.mean(pts2[:2], axis=1)
        mul2 = np.array([[np.sqrt(2) / np.std(pts2[:2]), 0, -np.sqrt(2) / np.std(pts2[:2]) * avg2[0]],
                   [0, np.sqrt(2) / np.std(pts2[:2]), -np.sqrt(2) / np.std(pts2[:2]) * avg2[1]],
                   [0, 0, 1]])
        pts1 = np.dot(mul1, pts1)
        pts2 = np.dot(mul2, pts2)

        A = np.zeros((pts1.shape[1], 9))
        for i in range(len):
            A[i] = [pts1[0, i] * pts2[0, i], pts1[0, i] * pts2[1, i], pts1[0, i] * pts2[2, i],
                pts1[1, i] * pts2[0, i], pts1[1, i] * pts2[1, i], pts1[1, i] * pts2[2, i],
                pts1[2, i] * pts2[0, i], pts1[2, i] * pts2[1, i], pts1[2, i] * pts2[2, i]]
        
        E=np.dot(A.T , A)
        M,N=np.linalg.eigh(E)
        F = N[:,0].reshape(3, 3).T
        X,Y=np.linalg.eig(np.dot(F,F.T))
        S=np.diag(np.sqrt(np.abs(X)))
        Z= temp.dot(F)
        
        S=np.diag(S)
        S[np.argmin(S)]=0
        S=np.diag(S)
        F=Y @ S @ Z
        F = mul2.T @ F @ mul1
        return  F/F[2,2]
    '''
    
    def poseMat(self, E):
        U, _, V = np.linalg.svd(E, full_matrices=True)
        X = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        #to compute the four projection matrices
        m1 = U[:, 2]
        m2 = -U[:, 2]
        m3 = U[:, 2]
        m4 = -U[:, 2]
        
        n1 = U @ X @ V
        n2 = n1
        n3 = U @ X.T @ V
        n4 = n3
        #check the determinant of that square array
        if np.linalg.det(n1) < 0:
            m1 = -m1 
            n1 = -n1
        if np.linalg.det(n2) < 0:
            m2 = -m2 
            n2 = -n2 
        if np.linalg.det(n3) < 0:
            m3 = -m3 
            n3 = -n3 
        if np.linalg.det(n4) < 0:
            m4 = -m4 
            n4 = -n4 
        m1 = m1.reshape((3,1))
        m2 = m2.reshape((3,1))
        m3 = m3.reshape((3,1))
        m4 = m4.reshape((3,1))
        return [n1, n2, n3, n4], [m1, m2, m3, m4]
    
    def calculatePose(self, totalRotmat, cameraMat, P1, P2):
        totalRotmatNum = len(totalRotmat)
        P1_Num = len(P1)
        limit = 0
        matrix_4d = np.identity(4)
        degree = 60
        
        for i in range(0, totalRotmatNum):
            
            Rmat = totalRotmat[i]
            value = math.sqrt(Rmat[0,0] * Rmat[0,0] + 
                        Rmat[1,0] * Rmat[1,0])
            #calculate the arc tangent from two vectors
            if  value < self.constant:
                angle1 = math.atan2(-Rmat[1,2], Rmat[1,1])
                angle2 = math.atan2(-Rmat[2,0], value)
                angle3 = 0
            else:
                angle1 = math.atan2(Rmat[2,1] , Rmat[2,2])
                angle2 = math.atan2(-Rmat[2,0], value)
                angle3 = math.atan2(Rmat[1,0], Rmat[0,0])
            #Convert degrees to radians
            AngleX = angle1 * (180/math.pi)
            AngleZ = angle3 * (180/math.pi)
            
            if AngleX < degree and AngleX > -degree:
                if AngleZ < degree and AngleZ > -degree:
                
                    total1 = 0
                    total2 = 0
                    combMat = np.hstack((totalRotmat[i], cameraMat[i]))
                    j = 0
                    while j < P1_Num:
                        
                        traingularPoint = self.getTrainPt(matrix_4d, combMat, P1[j], P2[j])
                        mix = traingularPoint - cameraMat[i]
                        rowZ = totalRotmat[i][2,:].reshape((1,3))
                        
                        decomMul = rowZ @ mix
                        if np.squeeze(decomMul) > 0: 
                            total1 += 1
                        else:
                            total2 += 1
                        j += 1
                    #iterate with the latest t and R, increase the limit
                    if total1 > limit:
                        limit = total1
                        t = cameraMat[i]
                        R = totalRotmat[i]
                    #print(total2)
                #print(angle3)
            #print(angle1)     
        if t[2] > 0:
            t = -t
            
        return R, t
    
    
    def getTrainPt(self, mat1, mat2, p1, p2):
        mat1 = mat1[0:3, :] 
        #matrix from P1
        prev = np.array([[0, -1, p1[1]], 
                    [1, 0, -p1[0]],
                    [-p1[1], p1[0], 0]])
        #matrix from P2
        prevprime = np.array([[0, -1, p2[1]],
                       [1, 0, -p2[0]],
                       [-p2[1], p2[0], 0]])
        
        M = prev @ mat1
        N = prevprime @ mat2
        mat3 = np.vstack((M, N))
        U, S, V = np.linalg.svd(mat3)
        
        X = (V[-1]/V[-1][3]).reshape((4,1))
        result = X[0:3]
        return result.reshape((3,1))

    
    def run(self):
        
        total_frame = len(self.frames)
        path = []
        #extract features by using fast algorithm
        P1 = self.fastfeature(self.imread(self.frames[0]))

        trans = np.array([[0],[0],[0]])
        rot = np.array([[1,0,0],[0,1,0],[0,0,1]])
        path.append([0, 0, 0])

        
        for i in range(1, total_frame):
            #use lucas-Kanade optical flow algorithm for matching
            P2, P1 = self.track(self.imread(self.frames[i-1]), self.imread(self.frames[i]), P1)
            
            if i % 50 == 0:
                print(i)
            
            #E = self.essentialMat_8_points(P1, P2)
            E, mask = cv2.findEssentialMat(P2, P1, self.k, cv2.RANSAC, prob=0.9999, threshold=1.5)
            #replace the cv2.recoverPose
            rotate_mat, cam_mat = self.poseMat(E)
            R, t = self.calculatePose(rotate_mat, cam_mat, P1, P2)
            
            #_, R, t, _ = cv2.recoverPose(E, P1, P2, self.k)

            scale = self.get_scale(i)
            trans = trans - scale * np.dot(rot,t)
            rot = np.dot(rot,R)
            path.append([trans[0][0], trans[1][0], trans[2][0]])
            P1 = self.fastfeature(self.imread(self.frames[i]))
        return np.array(path)



if __name__ == "__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path, path.shape)

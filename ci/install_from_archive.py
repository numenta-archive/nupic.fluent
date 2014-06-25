import sys
import requests
import tarfile

REGION = "us-west-2"
BUCKET = "artifacts.numenta.org"
REPO = "numenta/nupic"
SHA_FILE = "ci/nupic_sha.txt"



def fetchNupicTarballFor(sha):
  tarballName = "nupic-linux64-%s.tar.gz" % sha
  s3Url = "https://s3-%s.amazonaws.com/%s/%s/%s/%s" % (REGION, BUCKET, REPO, sha, tarballName)
  print "Fetching archive from %s..." % s3Url
  blockCount = 0
  with open(tarballName, "wb") as handle:
    response = requests.get(s3Url, stream=True)
    if not response.ok:
      raise Exception("Cannot fetch tarball from %s" % s3Url)
    for block in response.iter_content(1024):
      if blockCount % 100 == 0:
        sys.stdout.write('.')
        sys.stdout.flush()
      blockCount += 1
      if not block:
        break
      handle.write(block)
    print "\nDone."



def untar(filePath):
  print "Unzipping %s..." % filePath
  tarBall = tarfile.open(filePath, 'r:gz')
  tarBall.extractall('.')



def getSha():
  with open(SHA_FILE, "r") as shaFile:
    return shaFile.read().strip()



if __name__ == "__main__":
  sha = getSha()
  fetchNupicTarballFor(sha)
  untar("nupic-linux64-%s.tar.gz" % sha)
